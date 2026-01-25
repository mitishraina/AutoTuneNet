import torch
import random
from .base import NonStationarityRegime

class LabelNoiseShift(NonStationarityRegime):
    """
    Introduces label noise into the training dataset at a fixed epoch
    """
    def __init__(self, start_epoch: int, noise_ratio: float, seed: int = 42):
        super().__init__(start_epoch)
        self.noise_ratio = noise_ratio
        self.seed = seed

    def apply(self, train_dataset, **_):
        random.seed(self.seed)
        torch.manual_seed(self.seed)

        # Handle Subset datasets
        if hasattr(train_dataset, "dataset") and hasattr(train_dataset, "indices"):
            base_dataset = train_dataset.dataset
            indices = train_dataset.indices
        else:
            base_dataset = train_dataset
            indices = list(range(len(train_dataset)))

        # Access labels safely
        if hasattr(base_dataset, "targets"):
            labels = base_dataset.targets
        elif hasattr(base_dataset, "labels"):
            labels = base_dataset.labels
        else:
            raise AttributeError(
                "Dataset does not expose targets or labels"
            )

        num_classes = int(torch.max(labels).item() + 1)
        num_noisy = int(self.noise_ratio * len(indices))
        noisy_indices = random.sample(indices, num_noisy)

        for idx in noisy_indices:
            original_label = labels[idx]
            new_label = random.randint(0, num_classes - 1)
            while new_label == original_label:
                new_label = random.randint(0, num_classes - 1)
            labels[idx] = new_label

        print(
            f"[NON-STATIONARITY] Injected {self.noise_ratio:.0%} label noise "
            f"at epoch {self.start_epoch}"
        )
