"""
The Kaggle runtime: train an experiment's cells on Kaggle's GPU T4 x2 with
your own Kaggle account.

Stages (each skips what is already done, so any of them can be interrupted
and rerun):

- setup:   check the Kaggle CLI and your token (~/.kaggle/, never read here)
- data:    upload each task's data once, as a private dataset autotunenet-<task>
- prepare: build notebooks for the unfinished cells only (two GPUs each), saved
           under <results>/<experiment>/notebooks/ so you can read them
- start:   push the prepared notebooks and start them
- pull:    download finished notebooks' results into <results>/
- run:     all of the above, following progress until every notebook is done

Progress is kept in <results>/<experiment>/.runtime/kaggle.json (local only).
Notebooks install AutoTuneNet from the commit you are on when it is pushed to
GitHub, otherwise from PyPI at your installed version.
"""
import importlib.metadata
import json
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
from pathlib import Path
from typing import List

import autotunenet
from autotunenet.bench.run import pending_cells
from autotunenet.bench.runtimes.base import Runtime
from autotunenet.bench.tasks.data import DATA_DIR

JOB_TEMPLATE = Path(__file__).with_name("kaggle_job.py")
# enable_gpu alone gets "GPU T4 x2". Passing --accelerator NvidiaTeslaT4 explicitly sends
# batch runs to a separate queue where they can wait for hours.
MAX_GPU_SESSIONS = 2                      # Kaggle runs at most 2 batch GPU sessions per account
GPUS_PER_NOTEBOOK = 2
FINISHED = ("complete", "error", "cancelled")
TOKEN_HELP = ("Create a token at kaggle.com -> profile picture -> Settings -> API -> Create New Token. "
              "Kaggle saves it under ~/.kaggle/ (access_token, or a kaggle.json you move there and chmod 600).")


# ------------------------------------------------------------------ helpers

def kaggle(*args, check=True, capture=False):
    """Run the Kaggle CLI from this Python environment."""
    cli = Path(sys.executable).with_name("kaggle")
    if not cli.exists() and shutil.which("kaggle") is None:
        raise SystemExit('The Kaggle CLI is missing: pip install "autotunenet[bench]"')
    cmd = [str(cli) if cli.exists() else "kaggle", *args]
    return subprocess.run(cmd, check=check, text=True, capture_output=capture)


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def dataset_slug(task_name: str) -> str:
    return f"autotunenet-{slugify(task_name)}"


def install_spec() -> str:
    """What notebooks pip-install: the pushed commit of a git checkout, else the PyPI release."""
    repo = Path(autotunenet.__file__).resolve().parents[2]
    if (repo / ".git").exists():
        def git(*a):
            return subprocess.run(["git", "-C", str(repo), *a], text=True, capture_output=True).stdout.strip()
        commit = git("rev-parse", "HEAD")
        url = re.sub(r"^git@github\.com:", "https://github.com/", git("remote", "get-url", "origin"))
        git("fetch", "-q", "origin")
        if commit and url and git("branch", "-r", "--contains", commit):
            if git("status", "--porcelain", "--untracked-files=no", "--", "src"):
                print("[kaggle] note: uncommitted changes under src/ are not part of the run")
            return f"autotunenet[bench] @ git+{url}@{commit}"
        print(f"[kaggle] note: commit {commit[:8]} is not on GitHub; notebooks use the PyPI release")
    return f"autotunenet[bench]=={importlib.metadata.version('autotunenet')}"


def render_notebook(job: dict, title: str) -> dict:
    """kaggle_job.py -> notebook: `# %%` starts a code cell, `# %% [markdown]` a Markdown cell."""
    count = sum(len(g) for g in job["gpu_cells"])
    source = (JOB_TEMPLATE.read_text()
              .replace("__JOB__", json.dumps(job))
              .replace("__TITLE__", title)
              .replace("__EXPERIMENT__", job["experiment"])
              .replace("__INSTALL__", job["install"])
              .replace("__CELLS__", f"{count} ({' + '.join(str(len(g)) for g in job['gpu_cells'])} per GPU)"))
    cells = []
    for i, chunk in enumerate(re.split(r"^# %%", source, flags=re.M)[1:]):
        header, _, body = chunk.partition("\n")
        markdown = header.strip() == "[markdown]"
        lines = body.strip("\n").splitlines()
        if markdown:
            lines = [re.sub(r"^# ?", "", line) for line in lines]
        cells.append({"cell_type": "markdown" if markdown else "code", "id": f"cell-{i}", "metadata": {},
                      "source": "\n".join(lines).splitlines(keepends=True),
                      **({} if markdown else {"execution_count": None, "outputs": []})})
    return {"cells": cells, "nbformat": 4, "nbformat_minor": 5,
            "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                         "language_info": {"name": "python"}}}


# ------------------------------------------------------------------ runtime

class Kaggle(Runtime):
    name = "kaggle"

    def __init__(self):
        self._user = None

    # -- state ---------------------------------------------------------------
    @staticmethod
    def _state_path(exp, results_dir: Path) -> Path:
        return results_dir / exp.name / ".runtime" / "kaggle.json"

    def _load(self, exp, results_dir) -> dict:
        path = self._state_path(exp, results_dir)
        return json.loads(path.read_text()) if path.exists() else {"round": 0, "notebooks": []}

    def _save(self, exp, results_dir, state) -> None:
        path = self._state_path(exp, results_dir)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(state, indent=2))

    @property
    def user(self) -> str:
        if self._user is None:
            out = kaggle("config", "view", capture=True).stdout
            match = re.search(r"username:\s*(\S+)", out)
            if not match or match.group(1) == "None":
                raise SystemExit(f"No Kaggle account configured. {TOKEN_HELP}")
            self._user = match.group(1)
        return self._user

    def _state_of(self, kernel_id: str) -> str:
        out = kaggle("kernels", "status", kernel_id, check=False, capture=True)
        text = (out.stdout + out.stderr).lower()
        for state in ("complete", "error", "cancel", "running", "queued"):
            if state in text:
                return "cancelled" if state == "cancel" else state
        return "unknown"

    def _dataset_exists(self, task_name: str) -> bool:
        out = kaggle("datasets", "status", f"{self.user}/{dataset_slug(task_name)}", check=False, capture=True)
        return out.returncode == 0

    # -- stages --------------------------------------------------------------
    def setup(self) -> List[str]:
        problems = []
        try:
            print(f"[setup] Kaggle account: {self.user}")
        except SystemExit as e:
            problems.append(str(e))
        return problems

    def data(self, tasks, force=False):
        for task in tasks:
            if not task.data_paths:
                print(f"[data] {task.name}: no data files to upload")
                continue
            slug = dataset_slug(task.name)
            if self._dataset_exists(task.name) and not force:
                print(f"[data] {task.name}: already on Kaggle ({self.user}/{slug})")
                continue
            print(f"[data] {task.name}: uploading to {self.user}/{slug} (private)", flush=True)
            if task.prepare_data:
                task.prepare_data()
            with tempfile.TemporaryDirectory() as tmp:
                stage = Path(tmp)
                with tarfile.open(stage / "data.tar", "w") as tar:
                    for rel in task.data_paths:
                        src = DATA_DIR / rel
                        if not src.exists():
                            raise SystemExit(f"{task.name}: {src} not found; fetch the data first")
                        tar.add(src, arcname=rel)
                    marker = stage / "autotunenet_data.json"
                    marker.write_text(json.dumps({"task": task.name, "paths": list(task.data_paths)}))
                    tar.add(marker, arcname="autotunenet_data.json")
                marker.unlink()
                (stage / "dataset-metadata.json").write_text(json.dumps({
                    "title": f"AutoTuneNet data {task.name}"[:50], "id": f"{self.user}/{slug}",
                    "licenses": [{"name": "other"}]}))
                if self._dataset_exists(task.name):
                    kaggle("datasets", "version", "-p", str(stage), "-m", "update")
                else:
                    kaggle("datasets", "create", "-p", str(stage))
            print(f"[data] {task.name}: uploaded to {self.user}/{slug} (private)")

    def prepare(self, exp, exp_path, cells, results_dir, plugins, parallel):
        state = self._load(exp, results_dir)
        if any(not nb["pulled"] for nb in state["notebooks"]):
            print(f"[prepare] round {state['round']} is still in progress; continuing it")
            return
        todo = pending_cells(exp, results_dir, cells)
        if not todo:
            print("[prepare] every cell already has a result")
            return
        for spec in plugins:
            if not spec.endswith(".py"):
                raise SystemExit(f"--plugins {spec}: Kaggle needs plugin files (.py), not module names")

        if parallel > MAX_GPU_SESSIONS:
            print(f"[prepare] Kaggle runs at most {MAX_GPU_SESSIONS} GPU notebooks at once; "
                  f"using --parallel {MAX_GPU_SESSIONS}")
            parallel = MAX_GPU_SESSIONS
        groups = [[] for _ in range(parallel * GPUS_PER_NOTEBOOK)]
        for i, cell in enumerate(todo):
            groups[i % len(groups)].append(cell.to_dict())
        install = install_spec()
        notebooks_dir = results_dir / exp.name / "notebooks"
        notebooks_dir.mkdir(parents=True, exist_ok=True)
        notebooks = []
        count = sum(1 for k in range(parallel) if any(groups[k * GPUS_PER_NOTEBOOK + g]
                                                      for g in range(GPUS_PER_NOTEBOOK)))
        for k in range(count):
            gpu_cells = [groups[k * GPUS_PER_NOTEBOOK + g] for g in range(GPUS_PER_NOTEBOOK)]
            title = f"autotunenet {exp.name} {k + 1} of {count}"
            slug = slugify(title)
            tasks = sorted({c["task"] for g in gpu_cells for c in g})
            job = {"install": install, "experiment": exp.name, "experiment_yaml": Path(exp_path).read_text(),
                   "plugins": {Path(p).name: Path(p).read_text() for p in plugins}, "gpu_cells": gpu_cells}
            (notebooks_dir / f"{slug}.ipynb").write_text(json.dumps(render_notebook(job, title), indent=1))
            notebooks.append({"id": f"{self.user}/{slug}", "slug": slug, "title": title,
                              "datasets": [f"{self.user}/{dataset_slug(t)}" for t in tasks
                                           if self._dataset_exists(t)],
                              "cells": sum(len(g) for g in gpu_cells), "pushed": False, "pulled": False})
        state = {"round": state["round"] + 1, "install": install, "notebooks": notebooks}
        self._save(exp, results_dir, state)
        print(f"[prepare] round {state['round']}: {len(todo)} cells in {count} notebook(s), "
              f"saved to {notebooks_dir}")

    def start(self, exp, results_dir):
        state = self._load(exp, results_dir)
        notebooks_dir = results_dir / exp.name / "notebooks"
        for nb in state["notebooks"]:
            if nb["pushed"]:
                continue
            with tempfile.TemporaryDirectory() as tmp:
                folder = Path(tmp)
                shutil.copy(notebooks_dir / f"{nb['slug']}.ipynb", folder / "run.ipynb")
                (folder / "kernel-metadata.json").write_text(json.dumps({
                    "id": nb["id"], "title": nb["title"], "code_file": "run.ipynb", "language": "python",
                    "kernel_type": "notebook", "is_private": True, "enable_gpu": True, "enable_tpu": False,
                    "enable_internet": True, "dataset_sources": nb["datasets"], "competition_sources": [],
                    "kernel_sources": [], "model_sources": []}))
                kaggle("kernels", "push", "-p", str(folder), capture=True)
            nb["pushed"] = True
            self._save(exp, results_dir, state)
            print(f"[start] https://www.kaggle.com/code/{nb['id']} ({nb['cells']} cells)")

    def pull(self, exp, results_dir) -> int:
        state = self._load(exp, results_dir)
        added = 0
        for nb in state["notebooks"]:
            if not nb["pushed"] or nb["pulled"]:
                continue
            nb["state"] = self._state_of(nb["id"])
            if nb["state"] not in FINISHED:
                continue
            with tempfile.TemporaryDirectory() as tmp:
                kaggle("kernels", "output", nb["id"], "-p", tmp, check=False, capture=True)
                archive = Path(tmp) / "autotunenet_results.tar.gz"
                if archive.exists():
                    with tarfile.open(archive) as tar:
                        tar.extractall(tmp, filter="data")
                    src = Path(tmp) / "results"
                    for f in src.rglob("*.json") if src.exists() else []:
                        dest = results_dir / f.relative_to(src)
                        if not dest.exists():
                            dest.parent.mkdir(parents=True, exist_ok=True)
                            shutil.copy(f, dest)
                            added += 1
                    logs = results_dir / exp.name / ".runtime" / "logs" / nb["slug"]
                    if (Path(tmp) / "logs").exists():
                        shutil.copytree(Path(tmp) / "logs", logs, dirs_exist_ok=True)
            nb["pulled"] = True
            print(f"[pull] {nb['slug']}: {nb['state']}")
        self._save(exp, results_dir, state)
        return added

    def in_progress(self, exp, results_dir) -> bool:
        return any(nb["pushed"] and not nb["pulled"] for nb in self._load(exp, results_dir)["notebooks"])

    def describe(self, exp, results_dir) -> List[str]:
        state = self._load(exp, results_dir)
        if not state["notebooks"]:
            return ["kaggle: nothing prepared yet"]
        lines = [f"kaggle round {state['round']} ({state.get('install', '')})"]
        for nb in state["notebooks"]:
            where = ("downloaded" if nb["pulled"] else self._state_of(nb["id"]) if nb["pushed"] else "prepared")
            lines.append(f"  {nb['slug']}: {where}, {nb['cells']} cells")
        return lines

    def run(self, exp, exp_path, cells, results_dir, plugins, parallel, device):
        problems = self.setup()
        if problems:
            raise SystemExit("\n".join(problems))
        from autotunenet.bench.core.registry import TASKS
        todo = pending_cells(exp, results_dir, cells)
        self.data([TASKS.get(t) for t in sorted({c.task for c in todo})])
        self.prepare(exp, exp_path, cells, results_dir, plugins, parallel)
        self.start(exp, results_dir)
        started = time.time()
        while True:
            added = self.pull(exp, results_dir)
            left = len(pending_cells(exp, results_dir, cells))
            states = [l.strip() for l in self.describe(exp, results_dir)[1:]]
            print(f"[{(time.time() - started) / 60:5.1f} min] {len(cells) - left}/{len(cells)} cells done"
                  f"{f' (+{added})' if added else ''} | " + " | ".join(states), flush=True)
            if not self.in_progress(exp, results_dir):
                break
            time.sleep(60)
        return pending_cells(exp, results_dir, cells)
