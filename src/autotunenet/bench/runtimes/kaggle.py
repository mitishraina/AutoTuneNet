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
import queue
import threading
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
from autotunenet.bench import progress
from autotunenet.bench.run import pending_cells
from autotunenet.bench.runtimes.base import Runtime
from autotunenet.bench.tasks.data import DATA_DIR

JOB_TEMPLATE = Path(__file__).with_name("kaggle_job.py")
# enable_gpu alone gets "GPU T4 x2". Passing --accelerator NvidiaTeslaT4 explicitly sends
# batch runs to a separate queue where they can wait for hours.
MAX_GPU_SESSIONS = 2                      # Kaggle runs at most 2 batch GPU sessions per account
GPUS_PER_NOTEBOOK = 2
# "missing": the notebook was deleted on Kaggle. Its round ends, and the next run
# plans its unfinished cells again instead of waiting for it forever.
FINISHED = ("complete", "error", "cancelled", "missing")
TOKEN_HELP = ("Create a token at kaggle.com -> profile picture -> Settings -> API -> Create New Token. "
              "Kaggle saves it under ~/.kaggle/ (access_token, or a kaggle.json you move there and chmod 600).")


# ------------------------------------------------------------------ helpers

def _kaggle_cmd(*args) -> List[str]:
    cli = Path(sys.executable).with_name("kaggle")
    if not cli.exists() and shutil.which("kaggle") is None:
        raise SystemExit('The Kaggle CLI is missing: pip install "autotunenet[bench]"')
    return [str(cli) if cli.exists() else "kaggle", *args]


def kaggle(*args, check=True, capture=False):
    """Run the Kaggle CLI from this Python environment."""
    return subprocess.run(_kaggle_cmd(*args), check=check, text=True, capture_output=capture)


# Messages from the Kaggle CLI itself, not from the notebook. They must not end up in
# live.log, where they would look like errors of the run.
_CLI_MESSAGE = re.compile(r"^(\d{3} (Client|Server) Error|This is a problem on Kaggle's side|User cancelled operation)")


def _is_cli_message(line: str) -> bool:
    return bool(_CLI_MESSAGE.match(line.strip()))


def _clean(line: str) -> str:
    """Undo Kaggle's log stream decoding UTF-8 text as Latin-1 (e.g. 'Â·' for '·')."""
    try:
        return line.encode("latin-1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return line


class LogFollower(threading.Thread):
    """
    Follows a running notebook's live output (`kaggle kernels logs -f`), saves it
    to `live.log` and puts its progress events on a queue. Kaggle replays the
    output from the start on every connection; replayed events are ignored by
    `NotebookProgress.apply`.
    """

    def __init__(self, slug: str, kernel_id: str, log_file: Path, events: "queue.Queue"):
        super().__init__(daemon=True)
        self.slug, self.kernel_id, self.log_file, self.events = slug, kernel_id, log_file, events
        self.stopped = threading.Event()
        self.proc = None

    def run(self):
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        while not self.stopped.is_set():
            with open(self.log_file, "w") as log:
                self.proc = subprocess.Popen(_kaggle_cmd("kernels", "logs", "-f", self.kernel_id), text=True,
                                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1)
                for line in self.proc.stdout:
                    line = _clean(line)
                    if _is_cli_message(line):
                        continue
                    log.write(line)
                    log.flush()
                    event = progress.parse(line)
                    if event:
                        self.events.put(("event", self.slug, event))
                self.proc.wait()
            self.stopped.wait(20)                         # stream ended; reconnect while still running

    def stop(self):
        self.stopped.set()
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()


def snapshot(kernel_id: str, seconds: float = 20) -> List[dict]:
    """Events of a running notebook so far, read from its live output for a few seconds."""
    proc = subprocess.Popen(_kaggle_cmd("kernels", "logs", "-f", kernel_id), text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1)
    lines: List[str] = []
    reader = threading.Thread(target=lambda: lines.extend(proc.stdout), daemon=True)
    reader.start()
    reader.join(seconds)
    proc.terminate()
    return [e for e in map(progress.parse, (_clean(l) for l in lines if not _is_cli_message(l))) if e]


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
        if "cannot access kernel" in text or "not found" in text:
            return "missing"
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

    def pull(self, exp, results_dir, quiet: bool = False) -> int:
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
            if not quiet:
                print(f"[pull] {nb['slug']}: {nb['state']}")
        self._save(exp, results_dir, state)
        return added

    def in_progress(self, exp, results_dir) -> bool:
        return any(nb["pushed"] and not nb["pulled"] for nb in self._load(exp, results_dir)["notebooks"])

    def _board(self, exp, results_dir, state) -> "progress.Board":
        notebooks = [progress.NotebookProgress(title=nb["slug"], cells=nb["cells"],
                                               state="downloaded" if nb["pulled"] else nb.get("state", "prepared"))
                     for nb in state["notebooks"]]
        board = progress.Board(f"{exp.name} · Kaggle round {state['round']}", notebooks,
                               self._log_dir(exp, results_dir))
        return board

    @staticmethod
    def _log_dir(exp, results_dir) -> Path:
        return results_dir / exp.name / ".runtime" / "logs"

    def describe(self, exp, results_dir) -> List[str]:
        state = self._load(exp, results_dir)
        if not state["notebooks"]:
            return ["kaggle: nothing prepared yet"]
        board = self._board(exp, results_dir, state)
        board.live = False
        for nb, view in zip(state["notebooks"], board.notebooks):
            if nb["pushed"] and not nb["pulled"]:
                view.state = self._state_of(nb["id"])
                if view.state == "running":
                    live = board.log_dir / nb["slug"] / "live.log"
                    if live.exists() and time.time() - live.stat().st_mtime < 300:
                        events = [e for e in map(progress.parse, live.read_text().splitlines()) if e]
                    else:                                 # no `run` following it: read the stream briefly
                        events = snapshot(nb["id"])
                    for event in events:
                        view.apply(event)
        board.downloaded = sum(nb["cells"] for nb in state["notebooks"] if nb["pulled"])
        return board.lines()[2:]                         # skip the title and elapsed time

    def run(self, exp, exp_path, cells, results_dir, plugins, parallel, device):
        problems = self.setup()
        if problems:
            raise SystemExit("\n".join(problems))
        from autotunenet.bench.core.registry import TASKS
        todo = pending_cells(exp, results_dir, cells)
        self.data([TASKS.get(t) for t in sorted({c.task for c in todo})])
        self.prepare(exp, exp_path, cells, results_dir, plugins, parallel)
        self.start(exp, results_dir)
        return self.follow(exp, results_dir, cells)

    def follow(self, exp, results_dir, cells):
        """Show live progress until every notebook of the round is downloaded."""
        state = self._load(exp, results_dir)
        board = self._board(exp, results_dir, state)
        views = {nb["slug"]: view for nb, view in zip(state["notebooks"], board.notebooks)}
        events: "queue.Queue" = queue.Queue()
        followers: dict = {}
        board.downloaded = sum(nb["cells"] for nb in state["notebooks"] if nb["pulled"])
        next_poll = 0.0
        try:
            while True:
                if time.time() >= next_poll:
                    added = self.pull(exp, results_dir, quiet=True)
                    if added:
                        board.downloaded += added
                        board.note(f"downloaded {added} result(s)")
                    state = self._load(exp, results_dir)
                    for nb in state["notebooks"]:
                        view = views[nb["slug"]]
                        view.state = "downloaded" if nb["pulled"] else nb.get("state") or "prepared"
                        # Only a running notebook has a log to stream; a queued one returns an error.
                        running = nb["pushed"] and not nb["pulled"] and nb.get("state") == "running"
                        if running and nb["slug"] not in followers:
                            followers[nb["slug"]] = LogFollower(
                                nb["slug"], nb["id"], board.log_dir / nb["slug"] / "live.log", events)
                            followers[nb["slug"]].start()
                        if not running and nb["slug"] in followers:
                            followers.pop(nb["slug"]).stop()
                    if not self.in_progress(exp, results_dir):
                        break
                    next_poll = time.time() + 60
                while not events.empty():
                    kind, slug, event = events.get()
                    text = views[slug].apply(event) if kind == "event" else None
                    if text:
                        board.note(f"{slug} GPU {event.get('gpu', 0)}  {text}")
                board.draw()
                time.sleep(2)
        finally:
            for follower in followers.values():
                follower.stop()
        board.draw()
        if not board.live:
            print("\n".join(board.lines()), flush=True)
        return pending_cells(exp, results_dir, cells)
