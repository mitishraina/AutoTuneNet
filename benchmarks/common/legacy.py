"""
Shared setup for the legacy benchmark scripts in benchmarks/.

Every script accepts:
    --task   mnist | cifar10        (default mnist)
    --device auto | cpu | cuda | mps
    --seed   int                    (default 42)

and writes its outputs to benchmarks/<script>/results/<task>/seed_<seed>/.
Final accuracy is reported on the held-out test split, which no method uses
for tuning or model selection.
"""
import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

from autotunenet.factory import build_pytorch_autotunenet
from benchmarks.common.device import resolve_device
from benchmarks.common.tasks import (
    TASK_DEFAULTS,
    build_model,
    build_task,
    build_test_loader,
    resolve_task,
)
from benchmarks.common.train_utils import evaluate

EPOCHS = 10


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def autotunenet_config(task: str) -> dict:
    """AutoTuneNet settings, identical to the AutoTuneNet-Bench configs."""
    low, high = TASK_DEFAULTS[task]["lr_range"]
    return {
        "parameter_space": {"lr": {"low": low, "high": high, "log": True}},
        "search": {"n_startup_trials": 3},
        "tuning": {"tune_n_steps": 1, "warmup_epochs": 2, "max_delta": 0.5},
        "metrics": {"smoothing_window": 3},
        "stability": {"max_regression": 0.2, "patience": 2, "cooldown": 3},
    }


class LegacyRun:
    def __init__(self, script_file: str, description: str):
        parser = argparse.ArgumentParser(description)
        parser.add_argument("--task", default="mnist", choices=["mnist", "cifar10"])
        parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
        parser.add_argument("--seed", type=int, default=42)
        args = parser.parse_args()

        self.task = resolve_task(args.task)
        self.seed = args.seed
        self.device = resolve_device(args.device)
        self.defaults = TASK_DEFAULTS[self.task]
        self.batch_size = self.defaults["batch_size"]
        self.epochs = EPOCHS

        seed_everything(self.seed)
        self.train_loader, self.val_loader, _ = build_task(self.task, self.batch_size, self.seed)
        self.test_loader = build_test_loader(self.task, self.batch_size)

        self.out_dir = Path(script_file).parent / "results" / self.task / f"seed_{self.seed}"
        self.out_dir.mkdir(parents=True, exist_ok=True)
        print(f"[legacy] {Path(script_file).parent.name} task={self.task} seed={self.seed} device={self.device}")

    def new_model(self):
        """Fresh model; reseeds first so every method starts from the same weights."""
        seed_everything(self.seed)
        return build_model(self.task).to(self.device)

    def autotunenet_adapter(self, optimizer):
        return build_pytorch_autotunenet(optimizer, autotunenet_config(self.task), seed=self.seed)

    def test_metrics(self, model) -> dict:
        loss, acc = evaluate(model, self.test_loader, self.device)
        return {"test_loss": loss, "test_accuracy": acc}

    def run_info(self, **extra) -> dict:
        return {
            "task": self.task,
            "seed": self.seed,
            "device": self.device.type,
            "torch_version": torch.__version__,
            "batch_size": self.batch_size,
            **extra,
        }

    def save(self, name: str, obj) -> None:
        with open(self.out_dir / name, "w") as f:
            json.dump(obj, f, indent=2)
