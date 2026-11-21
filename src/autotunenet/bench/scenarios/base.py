"""Scenario: what happens to a training run besides the method's own decisions."""
from typing import Optional


class Scenario:
    """
    What happens to a training run besides the method's own decisions.

    Examples are label noise or gradient noise starting mid-run, or a bad
    initial learning rate. Register a subclass with
    `@SCENARIOS.register("name")`; its constructor keyword arguments come from
    the experiment file, and a `seed` argument receives the run's seed.

    Attributes:
        shift_epoch: First epoch trained under a shift, or `None` if there is none.
        overrides_initial_lr: `True` if every method, including population
            methods, must start from `initial_lr(task)`.
    """
    shift_epoch: Optional[int] = None
    overrides_initial_lr: bool = False   # True if every method must start from initial_lr(task)

    def initial_lr(self, task) -> float:
        """Learning rate every method starts from (default: the task's `lr`)."""
        return task.lr

    def on_epoch_start(self, epoch: int, trainer) -> None:
        """Called before each training epoch; may change the trainer's data or add hooks."""
        pass
