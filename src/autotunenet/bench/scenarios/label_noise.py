import random

import torch

from autotunenet.bench.core.registry import SCENARIOS
from autotunenet.bench.scenarios.base import Scenario


def unwrap_dataset(dataset):
    """
    Follow wrappers (Subset, transform wrappers) down to the base dataset.

    Returns (base_dataset, indices into it that belong to the given split).
    """
    base, indices = dataset, None
    while hasattr(base, "dataset"):
        if hasattr(base, "indices"):
            sub = list(base.indices)
            indices = sub if indices is None else [sub[i] for i in indices]
        base = base.dataset
    if indices is None:
        indices = list(range(len(base)))
    return base, indices


def corrupt_labels(dataset, noise_ratio: float, seed: int) -> int:
    """
    Replace the labels of `noise_ratio` of the split with a different class,
    chosen uniformly. Uses a local generator (global RNG untouched).
    Returns the number of labels changed.
    """
    rng = random.Random(seed)
    base, indices = unwrap_dataset(dataset)
    labels = getattr(base, "targets", None)
    if labels is None:
        labels = getattr(base, "labels", None)
    if labels is None:
        raise AttributeError("Dataset does not expose targets or labels")

    is_tensor = isinstance(labels, torch.Tensor)
    num_classes = int(labels.max().item() if is_tensor else max(labels)) + 1

    noisy = rng.sample(indices, int(noise_ratio * len(indices)))
    for idx in noisy:
        original = int(labels[idx])
        new_label = rng.randrange(num_classes - 1)   # uniform over the other classes
        if new_label >= original:
            new_label += 1
        labels[idx] = new_label
    return len(noisy)


@SCENARIOS.register("label_noise")
class LabelNoise(Scenario):
    """From `start_epoch`, a fraction of the training labels is randomised."""

    def __init__(self, start_epoch: int = 5, noise_ratio: float = 0.3, seed: int = 42):
        self.shift_epoch = start_epoch
        self.noise_ratio = noise_ratio
        self.seed = seed

    def on_epoch_start(self, epoch: int, trainer) -> None:
        if epoch == self.shift_epoch:
            corrupt_labels(trainer.train_dataset, self.noise_ratio, self.seed)
