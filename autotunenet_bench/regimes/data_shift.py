import random

import torch

from .base import NonStationarityRegime


class LabelNoiseShift(NonStationarityRegime):
    """
    Replaces the labels of a fraction of the training set with a different,
    uniformly chosen class, starting at a fixed epoch.

    Randomness comes from a local generator seeded with `seed`, so the global
    Python/torch RNG state (and with it data-loader shuffling) is unaffected.
    """
    def __init__(self, start_epoch: int, noise_ratio: float, seed: int = 42):
        super().__init__(start_epoch)
        self.noise_ratio = noise_ratio
        self.seed = seed

    @staticmethod
    def _unwrap(train_dataset):
        """Follow wrappers (Subset, transform wrappers) down to the base dataset.

        Returns (base_dataset, indices into the base dataset that belong to
        the training split).
        """
        base = train_dataset
        indices = None
        while hasattr(base, "dataset"):
            if hasattr(base, "indices"):
                sub = list(base.indices)
                indices = sub if indices is None else [sub[i] for i in indices]
            base = base.dataset
        if indices is None:
            indices = list(range(len(base)))
        return base, indices

    def apply(self, train_dataset, **_):
        rng = random.Random(self.seed)
        base, indices = self._unwrap(train_dataset)

        if hasattr(base, "targets"):
            labels = base.targets
        elif hasattr(base, "labels"):
            labels = base.labels
        else:
            raise AttributeError("Dataset does not expose targets or labels")

        is_tensor = isinstance(labels, torch.Tensor)
        num_classes = int(labels.max().item() if is_tensor else max(labels)) + 1

        num_noisy = int(self.noise_ratio * len(indices))
        noisy_indices = rng.sample(indices, num_noisy)

        for idx in noisy_indices:
            original = int(labels[idx])
            # Uniform over the other num_classes - 1 classes, without retries.
            new_label = rng.randrange(num_classes - 1)
            if new_label >= original:
                new_label += 1
            labels[idx] = new_label

        print(
            f"[NON-STATIONARITY] Injected {self.noise_ratio:.0%} label noise "
            f"at epoch {self.start_epoch}"
        )
