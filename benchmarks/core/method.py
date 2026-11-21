"""
Method interfaces.

Every method implements `Method.run(ctx, ledger) -> RunOutcome`.

Most methods train one model and only decide what to do at the end of each
epoch; they subclass SingleRunMethod and provide a small Controller. Methods
that train several models (offline HPO, PBT) implement run() themselves and
reuse ctx.new_trainer() / ctx.new_scenario().
"""
import inspect
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict

import torch

from benchmarks.core.ledger import ComputeLedger
from benchmarks.core.registry import SCENARIOS
from benchmarks.core.result import RunOutcome
from benchmarks.core.trainer import Trainer


@dataclass
class RunContext:
    task: Any                       # TaskSpec
    scenario_name: str
    scenario_params: Dict[str, Any]
    seed: int
    device: torch.device
    epochs: int
    options: Dict[str, Any] = field(default_factory=dict)

    def new_scenario(self):
        """
        A fresh scenario instance (each trained model needs its own).

        Scenarios that take a `seed` get the run seed unless the experiment
        pins one, so noise differs across seeds but is reproducible.
        """
        cls = SCENARIOS.get(self.scenario_name)
        params = dict(self.scenario_params)
        if "seed" in inspect.signature(cls).parameters and "seed" not in params:
            params["seed"] = self.seed
        return cls(**params)

    def new_trainer(self, seed: int | None = None) -> Trainer:
        return Trainer(self.task, self.seed if seed is None else seed, self.device)


class Method(ABC):
    """Base class; constructor keyword arguments come from the experiment file."""

    @abstractmethod
    def run(self, ctx: RunContext, ledger: ComputeLedger) -> RunOutcome:
        ...


class Controller(ABC):
    """Per-epoch decision logic of a single-run method."""

    @abstractmethod
    def on_epoch_end(self, epoch: int, val_loss: float) -> None:
        ...

    def extra(self) -> Dict[str, Any]:
        """Method-specific details to store with the result."""
        return {}


def train_single_run(ctx: RunContext, ledger: ComputeLedger, make_controller, trainer=None,
                     optimizer=None, scenario=None) -> RunOutcome:
    """
    Train one model for ctx.epochs with a controller acting after every epoch.

    Shared by every single-run method; also usable by multi-run methods to
    train one candidate.
    """
    scenario = scenario or ctx.new_scenario()
    trainer = trainer or ctx.new_trainer()
    optimizer = optimizer or torch.optim.Adam(trainer.model.parameters(), lr=scenario.initial_lr(ctx.task))
    controller = make_controller(optimizer, ctx)

    history = []
    for epoch in range(ctx.epochs):
        scenario.on_epoch_start(epoch, trainer)
        lr = optimizer.param_groups[0]["lr"]
        train_loss = trainer.train_epoch(optimizer)
        ledger.add_epochs(1)
        val_loss, val_accuracy = trainer.evaluate()
        history.append({"epoch": epoch, "lr": lr, "train_loss": train_loss,
                        "val_loss": val_loss, "val_accuracy": val_accuracy})
        with ledger.time_controller():
            controller.on_epoch_end(epoch, val_loss)

    test_loss, test_accuracy = trainer.test()
    return RunOutcome(test_loss, test_accuracy, history, controller.extra())


class SingleRunMethod(Method):
    """One model, ctx.epochs epochs, a Controller after each epoch."""

    @abstractmethod
    def make_controller(self, optimizer: torch.optim.Optimizer, ctx: RunContext) -> Controller:
        ...

    def run(self, ctx: RunContext, ledger: ComputeLedger) -> RunOutcome:
        return train_single_run(ctx, ledger, self.make_controller)
