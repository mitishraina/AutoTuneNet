from typing import Optional


class Scenario:
    """
    What happens to a training run besides the method's own decisions.

    shift_epoch: first epoch trained under a shift (None if there is none).
    initial_lr(task): learning rate every method starts from.
    on_epoch_start(epoch, trainer): called before each training epoch.
    """
    shift_epoch: Optional[int] = None

    def initial_lr(self, task) -> float:
        return task.lr

    def on_epoch_start(self, epoch: int, trainer) -> None:
        pass
