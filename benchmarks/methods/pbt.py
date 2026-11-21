"""
Population-based training (Jaderberg et al., 2017) via Ray Tune's
PopulationBasedTraining scheduler.

A population of models trains in parallel. Members share the run's
train/validation split but start from different initial weights and data
order (member 0 uses the run seed). Every `perturbation_interval`
epochs the weakest members (bottom `quantile_fraction`, by validation loss)
copy the weights and optimizer state of the strongest and perturb the learning
rate (x1.2 or x0.8, or resample it). The member with the best final validation
loss is reported.

Reproducibility: members run concurrently, as Ray's PBT is designed to. Which
results PBT compares at a synchronisation point can depend on scheduling
timing, so two runs with the same seed can differ slightly (all other methods
are bit-reproducible). Report PBT over several seeds. Running members one at
a time is not an option: Ray's synchronous PBT then stalls or never exploits.

Requires the `bench` dependency group (installed by `uv sync`):

    uv sync --group bench
"""
import atexit
import os
import random
import shutil
import tempfile
from pathlib import Path

import numpy as np
import torch

from benchmarks.core.method import Method, RunContext
from benchmarks.core.registry import METHODS
from benchmarks.core.result import RunOutcome

REPO_ROOT = Path(__file__).resolve().parents[2]
_STATE_FILE = "state.pt"
_DIVERGED = 1e9      # ranking value for a non-finite validation loss


def _save(directory, trainer, optimizer, epoch, history, owner):
    torch.save({**trainer.state(optimizer), "epoch": epoch, "history": history, "owner": owner,
                "torch_rng": torch.get_rng_state()},
               Path(directory) / _STATE_FILE)


def _load(checkpoint, device):
    with checkpoint.as_directory() as directory:
        return torch.load(Path(directory) / _STATE_FILE, map_location=device, weights_only=False)


def _log(path, *fields):
    """Append one line to the run's event log (workers share a machine; appends are atomic)."""
    with open(path, "a") as f:
        f.write(" ".join(str(x) for x in fields) + "\n")


def _train_member(config, ctx: RunContext, log_path: str, interval: int):
    """
    One population member, inside a Ray worker.

    Checkpoints are written only at perturbation boundaries (and at the end).
    Ray may let a member run past a boundary before pausing it; because the
    last checkpoint is the boundary one, a member that copies it always
    restarts exactly there, so every lineage trains exactly ctx.epochs epochs.
    Ray restarts members from a checkpoint at every boundary; it is an exploit
    when the checkpoint belongs to another member (logged to `log_path`).
    """
    from ray import tune

    from benchmarks.core.registry import load_plugins
    load_plugins()                                    # Ray workers are fresh processes: register built-ins

    me = tune.get_context().get_trial_id()
    trainer = ctx.new_trainer(init_seed=ctx.seed + 1000 * config["member"])
    scenario = ctx.new_scenario()
    optimizer = torch.optim.Adam(trainer.model.parameters(), lr=config["lr"])
    start_epoch, history = 0, []

    checkpoint = tune.get_checkpoint()
    if checkpoint is not None:
        state = _load(checkpoint, ctx.device)
        trainer.load_state(state, optimizer)
        for group in optimizer.param_groups:
            group["lr"] = config["lr"]                # possibly perturbed by PBT
        torch.set_rng_state(state["torch_rng"])       # data order continues, not restarts
        start_epoch, history = state["epoch"] + 1, state["history"]
        if state["owner"] != me:
            _log(log_path, "exploit", me, start_epoch)

        if start_epoch >= ctx.epochs:
            # Ray can exploit at the very last interval; nothing is left to
            # train, so report the copied state as this member's final result.
            with tempfile.TemporaryDirectory() as directory:
                _save(directory, trainer, optimizer, state["epoch"], history, owner=me)
                last = history[-1]["val_loss"]
                tune.report({"rank_loss": last if np.isfinite(last) else _DIVERGED,
                             "epochs_done": ctx.epochs},
                            checkpoint=tune.Checkpoint.from_directory(directory))
            return

    for epoch in range(start_epoch):                  # replay scenario events (label noise, hooks)
        scenario.on_epoch_start(epoch, trainer)

    for epoch in range(start_epoch, ctx.epochs):
        scenario.on_epoch_start(epoch, trainer)
        lr = optimizer.param_groups[0]["lr"]
        train_loss = trainer.train_epoch(optimizer)
        val_loss, val_accuracy = trainer.evaluate()
        history.append({"epoch": epoch, "lr": lr, "train_loss": train_loss,
                        "val_loss": val_loss, "val_accuracy": val_accuracy})
        metrics = {"rank_loss": val_loss if np.isfinite(val_loss) else _DIVERGED, "epochs_done": epoch + 1}
        if (epoch + 1) % interval == 0 or epoch + 1 == ctx.epochs:
            with tempfile.TemporaryDirectory() as directory:
                _save(directory, trainer, optimizer, epoch, history, owner=me)
                tune.report(metrics, checkpoint=tune.Checkpoint.from_directory(directory))
        else:
            tune.report(metrics)


@METHODS.register("pbt")
class PBT(Method):
    """Population-based training (Ray Tune). Defaults: population 4, exploit/explore every 2 epochs."""

    def __init__(self, population: int = 4, perturbation_interval: int = 2, quantile_fraction: float = 0.25,
                 resample_probability: float = 0.25):
        self.population = population
        self.perturbation_interval = perturbation_interval
        self.quantile_fraction = quantile_fraction
        self.resample_probability = resample_probability

    def run(self, ctx: RunContext, ledger) -> RunOutcome:
        try:
            import ray
            from ray import tune
            from ray.tune.schedulers import PopulationBasedTraining
        except ImportError as e:
            raise ImportError("PBT needs Ray Tune: run `uv sync --group bench`") from e

        low, high = ctx.task.lr_range
        scenario = ctx.new_scenario()
        # Members sample their starting LR, unless the scenario fixes it (stress test).
        initial_lr = scenario.initial_lr(ctx.task) if scenario.overrides_initial_lr else tune.loguniform(low, high)

        scheduler = PopulationBasedTraining(
            time_attr="epochs_done",
            perturbation_interval=self.perturbation_interval,
            hyperparam_mutations={"lr": tune.loguniform(low, high)},
            quantile_fraction=self.quantile_fraction,
            resample_probability=self.resample_probability,
            synch=True,                                   # all members move in lockstep: reproducible
        )

        random.seed(ctx.seed)                             # PBT's perturbation and sampling choices
        np.random.seed(ctx.seed)
        storage = tempfile.mkdtemp(prefix="pbt_")
        log_path = os.path.join(storage, "events.log")
        gpu = 1.0 / self.population if ctx.device.type == "cuda" else 0   # members share the GPU
        _ensure_ray(ray, self.population)
        try:
            tuner = tune.Tuner(
                tune.with_resources(tune.with_parameters(_train_member, ctx=ctx, log_path=log_path,
                                                         interval=self.perturbation_interval),
                                    {"cpu": 1, "gpu": gpu}),
                param_space={"lr": initial_lr, "member": tune.grid_search(list(range(self.population)))},
                tune_config=tune.TuneConfig(scheduler=scheduler, metric="rank_loss", mode="min",
                                            num_samples=1, max_concurrent_trials=self.population),
                run_config=tune.RunConfig(storage_path=storage, name="pbt", verbose=0),
            )
            results = tuner.fit()
            best = results.get_best_result(metric="rank_loss", mode="min", scope="last")
            state = _load(best.checkpoint, ctx.device)
            events = [line.split()[0] for line in open(log_path)] if os.path.exists(log_path) else []
        finally:
            shutil.rmtree(storage, ignore_errors=True)

        # Only the selected member is evaluated on the test split.
        trainer = ctx.new_trainer()
        trainer.model.load_state_dict(state["model"])
        test_loss, test_accuracy = trainer.test()

        # Every lineage trains exactly ctx.epochs epochs (exploited members
        # continue from the copied epoch), so the population trains
        # population * epochs; Ray's occasional re-run of a paused epoch is a
        # scheduling artefact, not part of the method's cost.
        ledger.add_epochs(self.population * ctx.epochs)
        ledger.note_parallel(self.population)
        ledger.trials = self.population
        return RunOutcome(test_loss, test_accuracy, state["history"], {
            "population": self.population,
            "exploits": events.count("exploit"),
            "final_lr": state["history"][-1]["lr"],
        })


def _ensure_ray(ray, population: int) -> None:
    """Start Ray once per process and reuse it for every PBT cell (starting it costs seconds)."""
    if ray.is_initialized():
        return
    env_vars = {"PYTHONPATH": str(REPO_ROOT)}
    if os.environ.get("AUTOTUNENET_DATA_DIR"):
        env_vars["AUTOTUNENET_DATA_DIR"] = os.environ["AUTOTUNENET_DATA_DIR"]
    ray.init(num_cpus=max(population, os.cpu_count() or 1), include_dashboard=False,
             log_to_driver=False, configure_logging=False, runtime_env={"env_vars": env_vars})
    atexit.register(ray.shutdown)
