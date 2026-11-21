"""
Run a benchmark experiment on Kaggle GPUs with your own Kaggle account.

One command does everything (upload data if missing, launch, wait with
progress, download results as notebooks finish, build the report):

    uv run python -m benchmarks.cloud.kaggle run --experiment benchmarks/experiments/comparison.yaml

Running it again while notebooks are still going attaches to them instead of
relaunching (use --relaunch to start over). The individual steps are also
available:

    uv run python -m benchmarks.cloud.kaggle upload-data                     # once: datasets -> private Kaggle dataset
    uv run python -m benchmarks.cloud.kaggle launch --experiment benchmarks/experiments/comparison.yaml --notebooks 3
    uv run python -m benchmarks.cloud.kaggle status --experiment comparison
    uv run python -m benchmarks.cloud.kaggle download --experiment comparison
    uv run python -m benchmarks.report --experiment comparison --plan benchmarks/experiments/comparison.yaml

Credentials: the Kaggle CLI reads your token from ~/.kaggle/ (access_token or
kaggle.json) or KAGGLE_* environment variables. This module never reads,
prints or stores them, and nothing account-specific is written to the repo.

Each notebook runs on Kaggle's "GPU T4 x2" and executes two shards at once,
one per GPU. The notebook clones this repository at the current commit, so
the commit must be pushed first. Finished cells are skipped on re-runs, so a
failed or timed-out notebook can simply be launched again.

The generated notebooks are saved locally so you can see exactly what ran:
by default under benchmarks/notebooks/local/ (git-ignored, for your own
experiments); with --record under benchmarks/notebooks/<experiment>/, which
is tracked, for the runs behind published results.
"""
import argparse
import json
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
from pathlib import Path

from benchmarks.core.experiment import Experiment
from benchmarks.core.registry import load_plugins
from benchmarks.tasks.data import DATA_DIR

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS = REPO_ROOT / "benchmarks" / "results"
JOB_TEMPLATE = Path(__file__).with_name("kaggle_job.py")
NOTEBOOKS = REPO_ROOT / "benchmarks" / "notebooks"
DATASET_SLUG = "autotunenet-data"
ACCELERATOR = "NvidiaTeslaT4"          # "GPU T4 x2"
GPUS_PER_NOTEBOOK = 2


# ----------------------------------------------------------------- helpers

def kaggle(*args, check=True, capture=False):
    """Run the Kaggle CLI from this environment."""
    cli = Path(sys.executable).with_name("kaggle")
    cmd = [str(cli) if cli.exists() else "kaggle", *args]
    return subprocess.run(cmd, check=check, text=True, capture_output=capture)


def git(*args):
    return subprocess.run(["git", "-C", str(REPO_ROOT), *args], check=True, text=True,
                          capture_output=True).stdout.strip()


def kaggle_username() -> str:
    out = kaggle("config", "view", capture=True).stdout
    match = re.search(r"username:\s*(\S+)", out)
    if not match or match.group(1) == "None":
        raise SystemExit("No Kaggle username found. Create an API token at kaggle.com -> Settings -> API "
                         "and save it under ~/.kaggle/ (see docs/BENCHMARKING.md).")
    return match.group(1)


def repo_url_and_commit():
    """The public clone URL and the current commit, which must already be on GitHub."""
    url = git("remote", "get-url", "origin")
    url = re.sub(r"^git@github\.com:", "https://github.com/", url)
    commit = git("rev-parse", "HEAD")
    git("fetch", "-q", "origin")
    if not git("branch", "-r", "--contains", commit):
        raise SystemExit(f"Commit {commit[:8]} is not on origin yet; push it first (the notebooks clone it).")
    if git("status", "--porcelain", "--untracked-files=no"):
        print("note: uncommitted changes are not included in the run")
    return url, commit


def manifest_path(experiment: str) -> Path:
    return RESULTS / experiment / ".kaggle_launch.json"


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def render_notebook(job: dict, title: str) -> dict:
    """
    Build the notebook from kaggle_job.py: `# %%` starts a code cell and
    `# %% [markdown]` a Markdown cell (its lines lose the leading `# `).
    """
    source = (JOB_TEMPLATE.read_text()
              .replace("__JOB__", json.dumps(job))
              .replace("__TITLE__", title)
              .replace("__EXPERIMENT__", job["experiment"])
              .replace("__COMMIT__", job["ref"])
              .replace("__SHARDS__", ", ".join(job["shards"])))
    cells = []
    for i, chunk in enumerate(re.split(r"^# %%", source, flags=re.M)[1:]):
        header, _, body = chunk.partition("\n")
        markdown = header.strip() == "[markdown]"
        lines = body.strip("\n").splitlines()
        if markdown:
            lines = [re.sub(r"^# ?", "", line) for line in lines]
        text = "\n".join(lines)
        cells.append({
            "cell_type": "markdown" if markdown else "code",
            "id": f"cell-{i}",
            "metadata": {},
            "source": text.splitlines(keepends=True),
            **({} if markdown else {"execution_count": None, "outputs": []}),
        })
    return {
        "cells": cells,
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                     "language_info": {"name": "python"}},
        "nbformat": 4,
        "nbformat_minor": 5,
    }


# ---------------------------------------------------------------- commands

def dataset_exists(user: str) -> bool:
    return kaggle("datasets", "status", f"{user}/{DATASET_SLUG}", check=False, capture=True).returncode == 0


def upload_data(args):
    """Upload MNIST and CIFAR-10 (downloading them locally first if needed) as a private dataset."""
    from torchvision import datasets

    from benchmarks.tasks.data import load_dataset

    for cls in (datasets.MNIST, datasets.CIFAR10):
        for train in (True, False):
            load_dataset(cls, train=train)

    user = kaggle_username()
    with tempfile.TemporaryDirectory() as tmp:
        stage = Path(tmp)
        raw = stage / "MNIST" / "raw"
        raw.mkdir(parents=True)
        for f in (DATA_DIR / "MNIST" / "raw").iterdir():
            if f.suffix != ".gz":                      # torchvision reads the uncompressed files
                shutil.copy(f, raw / f.name)
        shutil.copytree(DATA_DIR / "cifar-10-batches-py", stage / "cifar-10-batches-py")
        (stage / "dataset-metadata.json").write_text(json.dumps({
            "title": "AutoTuneNet benchmark data",
            "id": f"{user}/{DATASET_SLUG}",
            "licenses": [{"name": "other"}],
        }))
        if dataset_exists(user):
            kaggle("datasets", "version", "-p", str(stage), "-m", "update", "--dir-mode", "tar")
        else:
            kaggle("datasets", "create", "-p", str(stage), "--dir-mode", "tar")
    print(f"dataset: https://www.kaggle.com/datasets/{user}/{DATASET_SLUG} (private)")


def launch(args):
    load_plugins()
    exp_path = Path(args.experiment).resolve()
    exp = Experiment.load(exp_path)
    rel_exp = exp_path.relative_to(REPO_ROOT).as_posix()
    user = kaggle_username()
    repo, commit = repo_url_and_commit()

    shards = args.notebooks * GPUS_PER_NOTEBOOK
    has_data = dataset_exists(user)
    if not has_data:
        print("note: no data dataset found; notebooks will download MNIST/CIFAR-10 themselves "
              "(run `upload-data` once to avoid this)")
    filters = []
    for flag in ("method", "task", "scenario"):
        for value in getattr(args, flag) or []:
            filters += [f"--{flag}", value]

    cells = len(exp.cells(args.method, args.task, args.scenario))
    print(f"experiment {exp.name}: {cells} cells -> {shards} shards on {args.notebooks} notebook(s), "
          f"commit {commit[:8]}")

    saved = NOTEBOOKS / (exp.name if args.record else f"local/{exp.name}")
    saved.mkdir(parents=True, exist_ok=True)
    kernels = []
    for k in range(args.notebooks):
        own = [f"{k * GPUS_PER_NOTEBOOK + g + 1}/{shards}" for g in range(GPUS_PER_NOTEBOOK)]
        title = f"autotunenet {exp.name} {k + 1} of {args.notebooks}"
        slug = slugify(title)
        job = {"repo": repo, "ref": commit, "experiment": rel_exp, "shards": own, "filters": filters}
        notebook = json.dumps(render_notebook(job, title), indent=1)
        (saved / f"{slug}.ipynb").write_text(notebook)
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            (folder / "run.ipynb").write_text(notebook)
            (folder / "kernel-metadata.json").write_text(json.dumps({
                "id": f"{user}/{slug}",
                "title": title,
                "code_file": "run.ipynb",
                "language": "python",
                "kernel_type": "notebook",
                "is_private": True,
                "enable_gpu": True,
                "enable_tpu": False,
                "enable_internet": True,
                "dataset_sources": [f"{user}/{DATASET_SLUG}"] if has_data else [],
                "competition_sources": [],
                "kernel_sources": [],
                "model_sources": [],
            }, indent=2))
            kaggle("kernels", "push", "-p", str(folder), "--accelerator", ACCELERATOR)
        kernels.append({"id": f"{user}/{slug}", "shards": own})
        print(f"  launched https://www.kaggle.com/code/{user}/{slug}  (shards {', '.join(own)})")

    record = {"experiment": rel_exp, "commit": commit, "kernels": kernels}
    path = manifest_path(exp.name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2))
    (saved / "launch.json").write_text(json.dumps(record, indent=2))
    where = "tracked, commit it with the results" if args.record else "git-ignored"
    print(f"notebooks saved to {saved.relative_to(REPO_ROOT)} ({where})")
    print(f"\nCheck progress: uv run python -m benchmarks.cloud.kaggle status --experiment {exp.name}")


def _kernels(experiment):
    path = manifest_path(experiment)
    if not path.exists():
        raise SystemExit(f"No launch record for '{experiment}' ({path}); run `launch` first.")
    return json.loads(path.read_text())["kernels"]


def kernel_state(kernel_id: str) -> str:
    """'running', 'queued', 'complete', 'error', 'cancelled' or 'unknown'."""
    out = kaggle("kernels", "status", kernel_id, check=False, capture=True)
    text = (out.stdout + out.stderr).lower()
    for state in ("complete", "error", "cancel", "running", "queued"):
        if state in text:
            return "cancelled" if state == "cancel" else state
    return "unknown"


def status(args):
    for kernel in _kernels(args.experiment):
        print(f"{kernel['id']}: {kernel_state(kernel['id'])}")


def download(args, only=None):
    """Fetch finished notebooks' results into benchmarks/results/ (existing files are kept)."""
    added = 0
    for kernel in _kernels(args.experiment):
        if only is not None and kernel["id"] not in only:
            continue
        with tempfile.TemporaryDirectory() as tmp:
            out = kaggle("kernels", "output", kernel["id"], "-p", tmp, check=False, capture=True)
            archive = Path(tmp) / "autotunenet_results.tar.gz"
            if out.returncode != 0 or not archive.exists():
                print(f"{kernel['id']}: no results yet")
                continue
            with tarfile.open(archive) as tar:
                tar.extractall(tmp, filter="data")
            src = Path(tmp) / "results"
            for f in src.rglob("*.json") if src.exists() else []:
                dest = RESULTS / f.relative_to(src)
                if dest.exists() and not args.force:
                    continue
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(f, dest)
                added += 1
            logs = RESULTS / args.experiment / ".kaggle_logs" / kernel["id"].split("/")[-1]
            if (Path(tmp) / "logs").exists():
                shutil.copytree(Path(tmp) / "logs", logs, dirs_exist_ok=True)
            print(f"{kernel['id']}: results merged")
    print(f"{added} result file(s) added under {RESULTS}")
    return added


def _report(exp_path: Path, name: str):
    from benchmarks.report import build_report, load_plan
    if not list((RESULTS / name).glob("*/*/*/seed_*.json")):
        return None
    return build_report(RESULTS, name, "Support matrix", load_plan(exp_path))


def run(args):
    """Upload data if needed, launch (or attach), follow progress, download, report."""
    load_plugins()
    exp_path = Path(args.experiment).resolve()
    name = Experiment.load(exp_path).name
    user = kaggle_username()

    attach = manifest_path(name).exists() and not args.relaunch and any(
        kernel_state(k["id"]) in ("running", "queued") for k in _kernels(name))
    if attach:
        print(f"attaching to the running notebooks of '{name}' (use --relaunch to start over)")
    else:
        if not dataset_exists(user):
            upload_data(args)
        launch(args)

    kernels = _kernels(name)
    fetched, started = set(), time.time()
    while True:
        states = {k["id"]: kernel_state(k["id"]) for k in kernels}
        finished = [k for k, s in states.items() if s in ("complete", "error", "cancelled")]
        new = [k for k in finished if k not in fetched]
        if new:
            download(argparse.Namespace(experiment=name, force=False), only=set(new))
            fetched.update(new)
            out = _report(exp_path, name)
            if out:
                print(f"progress matrix: {out / 'support_matrix.svg'}")
        elapsed = (time.time() - started) / 60
        print(f"[{elapsed:5.1f} min] " + "  ".join(f"{k.split('/')[-1]}: {s}" for k, s in states.items()),
              flush=True)
        if len(finished) == len(kernels):
            break
        time.sleep(args.poll)

    out = _report(exp_path, name)
    failed = [k for k, s in states.items() if s != "complete"]
    if failed:
        print(f"\nnot completed: {', '.join(failed)} (rerun this command with --relaunch for the missing cells)")
    if out:
        print(f"\nreport: {out}")
        print((out / "support_matrix.md").read_text())


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run benchmark shards on Kaggle with your own account")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("upload-data", help="upload MNIST and CIFAR-10 as a private Kaggle dataset (once)")

    for name, helptext in (("run", "everything: upload data if missing, launch, follow, download, report"),
                           ("launch", "push one notebook per pair of shards and start them")):
        p = sub.add_parser(name, help=helptext)
        p.add_argument("--experiment", required=True, help="experiment YAML file in this repository")
        p.add_argument("--notebooks", type=int, default=3, help="notebooks to run (each runs 2 shards on 2 GPUs)")
        p.add_argument("--record", action="store_true",
                       help="save the notebooks under benchmarks/notebooks/<experiment>/ (tracked) instead of "
                            "benchmarks/notebooks/local/ (git-ignored); use for runs behind published results")
        for flag in ("method", "task", "scenario"):
            p.add_argument(f"--{flag}", action="append", help=f"only this {flag} (repeatable)")
        if name == "run":
            p.add_argument("--relaunch", action="store_true", help="start new notebooks even if some are running")
            p.add_argument("--poll", type=int, default=60, help="seconds between progress checks")

    for name, helptext in (("status", "show each notebook's state"),
                           ("download", "merge finished notebooks' results into benchmarks/results/")):
        p = sub.add_parser(name, help=helptext)
        p.add_argument("--experiment", required=True, help="experiment name (e.g. comparison)")
        if name == "download":
            p.add_argument("--force", action="store_true", help="overwrite results that already exist locally")

    args = parser.parse_args(argv)
    {"upload-data": upload_data, "run": run, "launch": launch, "status": status,
     "download": download}[args.command](args)


if __name__ == "__main__":
    main()
