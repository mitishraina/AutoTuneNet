"""
The script every Kaggle notebook runs (uploaded as the notebook's code).

It clones the repository at a fixed commit, installs the locked environment,
points the benchmarks at the attached dataset (or downloads the data if none
is attached), runs one shard per GPU in parallel, and packs the results into
/kaggle/working/autotunenet_results.tar.gz for `kaggle kernels output`.
The launcher fills in the JOB dictionary below.
"""
import glob
import json
import os
import shutil
import subprocess
import tarfile
import time

JOB = json.loads(r'''__JOB__''')

WORK = "/kaggle/temp/autotunenet"          # not part of the notebook output
REPO_DIR = os.path.join(WORK, "repo")
RESULTS = "/kaggle/working/results"
LOGS = "/kaggle/working/logs"


def sh(cmd, **kwargs):
    print(f"$ {cmd}", flush=True)
    subprocess.run(cmd, shell=True, check=True, **kwargs)


def find_data_dir():
    """Directory holding MNIST/ and cifar-10-batches-py/ from an attached dataset, if any."""
    for root in glob.glob("/kaggle/input/**/", recursive=True):
        if os.path.isdir(os.path.join(root, "cifar-10-batches-py")) or os.path.isdir(os.path.join(root, "MNIST")):
            return root.rstrip("/")
    for archive in glob.glob("/kaggle/input/**/*.tar", recursive=True):
        target = os.path.join(WORK, "data")
        os.makedirs(target, exist_ok=True)
        with tarfile.open(archive) as tar:
            tar.extractall(target)
        return target
    return None


def main():
    start = time.time()
    os.makedirs(WORK, exist_ok=True)
    os.makedirs(LOGS, exist_ok=True)
    sh("pip install -q uv")
    sh(f"git clone -q {JOB['repo']} {REPO_DIR}")
    sh(f"git -C {REPO_DIR} checkout -q {JOB['ref']}")
    sh("uv sync --locked --quiet", cwd=REPO_DIR)

    env = dict(os.environ)
    data = find_data_dir()
    env["AUTOTUNENET_DATA_DIR"] = data or os.path.join(WORK, "data")
    print(f"data: {env['AUTOTUNENET_DATA_DIR']} ({'attached dataset' if data else 'downloaded on first use'})")
    sh("nvidia-smi --query-gpu=index,name,memory.total --format=csv")

    procs = []
    for gpu, shard in enumerate(JOB["shards"]):
        cmd = ["uv", "run", "python", "-m", "benchmarks.run", "--experiment", JOB["experiment"],
               "--shard", shard, "--device", "cuda", "--results-dir", RESULTS] + JOB["filters"]
        log = open(os.path.join(LOGS, f"shard_{shard.replace('/', '_of_')}.log"), "w")
        print(f"starting shard {shard} on GPU {gpu}", flush=True)
        procs.append((shard, subprocess.Popen(cmd, cwd=REPO_DIR, env={**env, "CUDA_VISIBLE_DEVICES": str(gpu)},
                                              stdout=log, stderr=subprocess.STDOUT)))
    failed = [shard for shard, p in procs if p.wait() != 0]

    with tarfile.open("/kaggle/working/autotunenet_results.tar.gz", "w:gz") as tar:
        if os.path.isdir(RESULTS):
            tar.add(RESULTS, arcname="results")
        tar.add(LOGS, arcname="logs")
    shutil.rmtree(RESULTS, ignore_errors=True)
    print(f"done in {(time.time() - start) / 60:.1f} min; failed shards: {failed or 'none'}")
    for shard, _ in procs:
        path = os.path.join(LOGS, f"shard_{shard.replace('/', '_of_')}.log")
        print(f"--- tail of shard {shard} ---")
        print("".join(open(path).readlines()[-15:]))


if __name__ == "__main__":
    main()
