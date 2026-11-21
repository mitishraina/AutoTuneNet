"""
Download the benchmark datasets ahead of time.

Usage:
    uv run python -m benchmarks.download_data             # all datasets
    uv run python -m benchmarks.download_data --task cifar10

Datasets go to <repo>/data (or $AUTOTUNENET_DATA_DIR). Anything already
there is reused. Benchmarks also download what they need automatically, so
this step is optional.
"""
import argparse

from torchvision import datasets

from benchmarks.tasks.data import DATA_DIR, load_dataset

TASK_DATASETS = {"mnist": datasets.MNIST, "cifar10": datasets.CIFAR10}


def main():
    parser = argparse.ArgumentParser(description="Download the benchmark datasets")
    parser.add_argument("--task", choices=sorted(TASK_DATASETS), action="append",
                        help="Dataset to download; repeat for several. Default: all.")
    args = parser.parse_args()

    tasks = args.task or sorted(TASK_DATASETS)
    print(f"[data] Data directory: {DATA_DIR}")
    for task in tasks:
        for train in (True, False):
            load_dataset(TASK_DATASETS[task], train=train)
        print(f"[data] {task}: training and test splits available")


if __name__ == "__main__":
    main()
