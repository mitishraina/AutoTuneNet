import random
from types import SimpleNamespace

import torch
from torch.utils.data import Subset

from autotunenet.bench.core.trainer import Trainer
from autotunenet.bench.core.registry import TASKS
from autotunenet.bench.scenarios.bad_initial_lr import BadInitialLR
from autotunenet.bench.scenarios.gradient_noise import GradientNoise
from autotunenet.bench.scenarios.label_noise import corrupt_labels, LabelNoise
from autotunenet.bench.tasks.toy import LabelledTensors


def _labels(n=1000, tensor=False):
    targets = torch.tensor([i % 10 for i in range(n)]) if tensor else [i % 10 for i in range(n)]
    return LabelledTensors(torch.zeros(n, 1), targets)


# --------------------------------------------------------------- label noise

def test_label_noise_changes_the_right_share_of_the_split_only():
    base = _labels()
    train_idx = list(range(0, 1000, 2))
    before = list(base.targets)
    assert corrupt_labels(Subset(base, train_idx), 0.3, seed=7) == 150
    changed = [i for i in range(1000) if before[i] != base.targets[i]]
    assert len(changed) == 150 and set(changed) <= set(train_idx)


def test_label_noise_works_with_tensor_targets_and_keeps_global_rng():
    random.seed(3); torch.manual_seed(3)
    expected = (random.random(), torch.rand(1).item())
    random.seed(3); torch.manual_seed(3)
    base = _labels(200, tensor=True)
    before = base.targets.clone()
    corrupt_labels(base, 0.3, seed=1)
    assert int((before != base.targets).sum()) == 60
    assert (random.random(), torch.rand(1).item()) == expected


def test_label_noise_scenario_applies_once_at_the_shift():
    trainer = SimpleNamespace(train_dataset=_labels(100))
    scenario = LabelNoise(start_epoch=2, noise_ratio=0.5, seed=1)
    before = list(trainer.train_dataset.targets)
    scenario.on_epoch_start(1, trainer)
    assert trainer.train_dataset.targets == before
    scenario.on_epoch_start(2, trainer)
    assert sum(a != b for a, b in zip(before, trainer.train_dataset.targets)) == 50


# ------------------------------------------------------------ gradient noise

def _train(scenario=None, epochs=3):
    trainer = Trainer(TASKS.get("toy"), 0, torch.device("cpu"))
    optimizer = torch.optim.SGD(trainer.model.parameters(), lr=0.1)
    snapshots = []
    for epoch in range(epochs):
        if scenario:
            scenario.on_epoch_start(epoch, trainer)
        trainer.train_epoch(optimizer)
        snapshots.append(torch.cat([p.detach().flatten().clone() for p in trainer.model.parameters()]))
    return snapshots


def test_gradient_noise_only_after_the_shift_and_reproducible():
    clean = _train()
    noisy = _train(GradientNoise(start_epoch=1, std=0.5, seed=7))
    again = _train(GradientNoise(start_epoch=1, std=0.5, seed=7))
    assert torch.equal(clean[0], noisy[0])
    assert not torch.equal(clean[-1], noisy[-1])
    assert torch.equal(noisy[-1], again[-1])


def test_gradient_noise_hooks_can_be_removed():
    trainer = Trainer(TASKS.get("toy"), 0, torch.device("cpu"))
    scenario = GradientNoise(start_epoch=0, std=0.5)
    scenario.on_epoch_start(0, trainer)
    scenario.remove()
    trainer.model(torch.ones(1, 2)).sum().backward()
    grads_clean = [p.grad.clone() for p in trainer.model.parameters()]
    trainer.model.zero_grad()
    trainer.model(torch.ones(1, 2)).sum().backward()
    assert all(torch.equal(a, p.grad) for a, p in zip(grads_clean, trainer.model.parameters()))


# ------------------------------------------------------------ bad initial lr

def test_bad_initial_lr_overrides_the_task_default():
    assert BadInitialLR(initial_lr=0.3).initial_lr(TASKS.get("toy")) == 0.3
