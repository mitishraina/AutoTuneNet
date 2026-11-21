# Extending the benchmark

You can benchmark any dataset and model, add a scenario, or add a method without changing the package. Write them in a Python file, register them, and pass the file with `--plugins` (repeatable) to any command. On Kaggle the same file is shipped inside the notebooks, and PBT's Ray workers load it too.

```bash
autotunenet list --plugins examples/bench_plugin.py
autotunenet run --experiment examples/bench_experiment.yaml --plugins examples/bench_plugin.py
autotunenet run --experiment examples/bench_experiment.yaml --plugins examples/bench_plugin.py --runtime kaggle
```

[`examples/bench_plugin.py`](https://github.com/mitishraina/AutoTuneNet/blob/main/examples/bench_plugin.py) is a complete, tested plugin: a Fashion-MNIST task and an input-noise scenario. [`examples/bench_experiment.yaml`](https://github.com/mitishraina/AutoTuneNet/blob/main/examples/bench_experiment.yaml) runs it against four methods.

## A task (any dataset and model)

A task is a `TaskSpec`: how to build the data loaders, the test loader and a fresh model, plus training defaults.

```python
--8<-- "examples/bench_plugin.py:task"
```

| Field | Meaning |
|---|---|
| `build_loaders(batch_size, seed)` | Train and validation loaders. Split with `seed`, so every method sees the same split |
| `build_test_loader(batch_size)` | Held-out test loader, used once per run for the reported numbers |
| `build_model()` | A freshly initialised model; the harness seeds it |
| `batch_size`, `lr` | Training defaults; `lr` is where every single-run method starts |
| `lr_range` | The log-scale range every tuning method searches |
| `data_paths` | Files or folders under the data directory that the task reads (needed for hosted runtimes) |
| `prepare_data` | Downloads them into the data directory if missing (`autotunenet data` calls it) |

Read data from the data directory (`autotunenet.bench.paths.data_dir()`: `./data` or `$AUTOTUNENET_DATA_DIR`). `autotunenet data --runtime kaggle` uploads `data_paths` as a private dataset, and the notebook rebuilds the same directory, so your loader code runs there unchanged. For torchvision datasets, `autotunenet.bench.tasks.data.load_dataset` handles the download, with a progress bar.

## A scenario

A scenario changes training independently of the method: corrupt data, add noise, or force a starting learning rate. Subclass `Scenario` and register it:

```python
--8<-- "examples/bench_plugin.py:scenario"
```

| Member | Meaning |
|---|---|
| `shift_epoch` | First epoch trained under the change, or `None`. Used for post-shift metrics |
| `on_epoch_start(epoch, trainer)` | Called before every epoch. `trainer.model`, `trainer.train_dataset` and `trainer.train_loader` can be changed |
| `initial_lr(task)` | The learning rate every method starts from (default `task.lr`) |
| `overrides_initial_lr` | Set `True` to make population methods start from `initial_lr` too |

Constructor arguments come from the experiment file. An argument named `seed` receives the run's seed. Methods that restore checkpoints replay `on_epoch_start` for the skipped epochs, so a scenario must apply its change in `on_epoch_start` and keep no other state that would be lost.

## A method

Most methods train one model and decide something at the end of each epoch. Subclass `SingleRunMethod` and return a `Controller`:

```python
import torch

from autotunenet.bench.core.method import Controller, SingleRunMethod
from autotunenet.bench.core.registry import METHODS


class _StepDecay(Controller):
    def __init__(self, optimizer, step_size, gamma):
        self.scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size, gamma)

    def on_epoch_end(self, epoch, val_loss):
        self.scheduler.step()


@METHODS.register("step_decay")
class StepDecay(SingleRunMethod):
    """Multiply the learning rate by `gamma` every `step_size` epochs."""

    def __init__(self, step_size: int = 3, gamma: float = 0.5):
        self.step_size, self.gamma = step_size, gamma

    def make_controller(self, optimizer, ctx):
        return _StepDecay(optimizer, self.step_size, self.gamma)
```

To make it built in instead, put the module in `src/autotunenet/bench/methods/`, import it in that package's `__init__.py`, add it (with its arguments) to `experiments/smoke.yaml`, and run `uv run pytest`. The end-to-end test fails if a registered method is missing from `smoke.yaml`, and runs it under every scenario on the toy task.

Methods that train several models (offline HPO, population-based training) subclass `Method` and implement `run(ctx, ledger)`. They create models with `ctx.new_trainer()` and `ctx.new_scenario()`, train each candidate with `train_single_run(...)` or the `Trainer` directly, record the cost in the ledger, and return a `RunOutcome` for the model they report.

