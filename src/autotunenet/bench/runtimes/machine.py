"""The default runtime: train on this machine (CUDA GPU, then Apple GPU, then CPU)."""
from typing import List

from autotunenet.bench.core.device import resolve_device
from autotunenet.bench.run import execute, pending_cells
from autotunenet.bench.runtimes.base import Runtime


class Machine(Runtime):
    name = "machine"

    def setup(self) -> List[str]:
        print(f"[setup] this machine: training on {resolve_device('auto')}")
        return []

    def data(self, tasks, force=False):
        for task in tasks:
            if task.prepare_data:
                task.prepare_data()
            print(f"[data] {task.name}: ready")

    def prepare(self, exp, exp_path, cells, results_dir, plugins, parallel):
        print(f"[prepare] {len(pending_cells(exp, results_dir, cells))} cells to run on this machine")

    def start(self, exp, results_dir):
        print("[start] nothing to start: `autotunenet run` trains the cells here")

    def run(self, exp, exp_path, cells, results_dir, plugins, parallel, device):
        execute(exp, cells, device, results_dir)
        return pending_cells(exp, results_dir, cells)
