"""
Run a benchmark experiment on Kaggle GPUs with your own Kaggle account.

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
"""
import argparse
import json
import math
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

from benchmarks.core.experiment import Experiment
from benchmarks.core.registry import load_plugins
from benchmarks.tasks.data import DATA_DIR

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS = REPO_ROOT / "benchmarks" / "results"
JOB_TEMPLATE = Path(__file__).with_name("kaggle_job.py")
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


# ---------------------------------------------------------------- commands

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
        exists = kaggle("datasets", "status", f"{user}/{DATASET_SLUG}", check=False, capture=True).returncode == 0
        if exists:
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
    has_data = kaggle("datasets", "status", f"{user}/{DATASET_SLUG}", check=False, capture=True).returncode == 0
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

    kernels = []
    template = JOB_TEMPLATE.read_text()
    for k in range(args.notebooks):
        own = [f"{k * GPUS_PER_NOTEBOOK + g + 1}/{shards}" for g in range(GPUS_PER_NOTEBOOK)]
        title = f"autotunenet {exp.name} {k + 1} of {args.notebooks}"
        slug = slugify(title)
        job = {"repo": repo, "ref": commit, "experiment": rel_exp, "shards": own, "filters": filters}
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            (folder / "run.py").write_text(template.replace("__JOB__", json.dumps(job)))
            (folder / "kernel-metadata.json").write_text(json.dumps({
                "id": f"{user}/{slug}",
                "title": title,
                "code_file": "run.py",
                "language": "python",
                "kernel_type": "script",
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

    path = manifest_path(exp.name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"experiment": rel_exp, "commit": commit, "kernels": kernels}, indent=2))
    print(f"\nCheck progress: uv run python -m benchmarks.cloud.kaggle status --experiment {exp.name}")


def _kernels(experiment):
    path = manifest_path(experiment)
    if not path.exists():
        raise SystemExit(f"No launch record for '{experiment}' ({path}); run `launch` first.")
    return json.loads(path.read_text())["kernels"]


def status(args):
    for kernel in _kernels(args.experiment):
        out = kaggle("kernels", "status", kernel["id"], check=False, capture=True)
        line = (out.stdout or out.stderr).strip().splitlines()
        print(f"{kernel['id']}: {line[-1] if line else 'unknown'}")


def download(args):
    """Fetch every finished notebook's results into benchmarks/results/ (existing files are kept)."""
    added = 0
    for kernel in _kernels(args.experiment):
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


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run benchmark shards on Kaggle with your own account")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("upload-data", help="upload MNIST and CIFAR-10 as a private Kaggle dataset (once)")

    p = sub.add_parser("launch", help="push one notebook per pair of shards and start them")
    p.add_argument("--experiment", required=True, help="experiment YAML file in this repository")
    p.add_argument("--notebooks", type=int, default=3, help="notebooks to run (each runs 2 shards on 2 GPUs)")
    for flag in ("method", "task", "scenario"):
        p.add_argument(f"--{flag}", action="append", help=f"only this {flag} (repeatable)")

    for name, helptext in (("status", "show each notebook's state"),
                           ("download", "merge finished notebooks' results into benchmarks/results/")):
        p = sub.add_parser(name, help=helptext)
        p.add_argument("--experiment", required=True, help="experiment name (e.g. comparison)")
        if name == "download":
            p.add_argument("--force", action="store_true", help="overwrite results that already exist locally")

    args = parser.parse_args(argv)
    {"upload-data": upload_data, "launch": launch, "status": status, "download": download}[args.command](args)


if __name__ == "__main__":
    main()
