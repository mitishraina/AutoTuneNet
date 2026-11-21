import random

import optuna
import torch
from torch.utils.data import DataLoader, Subset, TensorDataset

from autotunenet.factory import build_pytorch_autotunenet
from autotunenet_bench.controllers.autotunenet import AutoTuneNetController
from autotunenet_bench.metrics.compute_metrics import ComputeMetrics
from autotunenet_bench.regimes.data_shift import LabelNoiseShift
from benchmarks.common.train_utils import evaluate

optuna.logging.set_verbosity(optuna.logging.ERROR)


# ------------------------------------------------------------- controller (M7)

def _controller(warmup=0):
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    adapter = build_pytorch_autotunenet(optimizer, {
        "parameter_space": {"lr": {"low": 1e-4, "high": 1e-1, "log": True}},
        "tuning": {"warmup_epochs": warmup},
        "stability": {"max_regression": 0.1, "patience": 1, "cooldown": 0},
    }, seed=0)
    return AutoTuneNetController(adapter)


def test_controller_reports_trial_starts_and_updates():
    ctl = _controller(warmup=1)
    events = []
    for metric in [-0.1, -0.1, -0.1, -5.0]:
        ctl.on_epoch_end(metric)
        events.append((ctl.last_trial_start, ctl.last_update, ctl.last_rollback))
    assert events == [
        (False, False, False),   # warmup
        (True, True, False),     # first proposal applied
        (True, True, False),     # scored, next proposal applied
        (False, True, True),     # regression -> rollback changes values back
    ]


def _simulate(events):
    """Drive ComputeMetrics the way train_runner does."""
    m = ComputeMetrics()
    for trial_start, update, rollback in events:
        m.step_epoch()
        if update:
            m.record_update()
        if rollback:
            m.record_rollback()
            m.end_trial()
        if trial_start:
            m.start_trial()
    return m.summary()


def test_compute_metrics_count_trials_and_their_length():
    s = _simulate([
        (False, False, False),   # epoch 0: warmup
        (True, True, False),     # epoch 1: trial A starts
        (False, False, False),   # epoch 2: A runs
        (True, True, False),     # epoch 3: A ran 2 epochs; trial B starts
        (False, True, True),     # epoch 4: B ran 1 epoch, rolled back
    ])
    assert s["num_trials_started"] == 2
    assert s["num_hparam_updates"] == 3
    assert s["num_rollbacks"] == 1
    assert s["epochs_per_trial_avg"] == 1.5


def test_trial_still_running_at_the_end_is_counted():
    s = _simulate([(True, True, False), (False, False, False), (False, False, False)])
    assert s["epochs_per_trial_avg"] == 2


# ----------------------------------------------------------- label noise (L3)

class _Labelled(torch.utils.data.Dataset):
    def __init__(self, targets):
        self.targets = targets

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, i):
        return torch.zeros(1), self.targets[i]


def _noisy(targets, train_idx, ratio=0.3, seed=7):
    base = _Labelled(targets)
    before = [int(t) for t in targets]
    LabelNoiseShift(0, ratio, seed=seed).apply(Subset(base, train_idx))
    after = [int(t) for t in base.targets]
    return before, after


def test_label_noise_changes_the_right_share_of_training_labels_only():
    targets = [i % 10 for i in range(1000)]
    train_idx = list(range(0, 1000, 2))
    before, after = _noisy(list(targets), train_idx)
    changed = [i for i in range(1000) if before[i] != after[i]]
    assert len(changed) == 150                       # 30% of 500
    assert set(changed) <= set(train_idx)            # validation labels untouched
    assert all(0 <= after[i] < 10 for i in changed)


def test_label_noise_works_with_tensor_targets():
    targets = torch.tensor([i % 10 for i in range(200)])
    before, after = _noisy(targets, list(range(200)))
    assert sum(b != a for b, a in zip(before, after)) == 60


def test_label_noise_leaves_the_global_rng_alone():
    random.seed(3)
    torch.manual_seed(3)
    expected = (random.random(), torch.rand(1).item())
    random.seed(3)
    torch.manual_seed(3)
    _noisy([i % 10 for i in range(100)], list(range(100)))
    assert (random.random(), torch.rand(1).item()) == expected


def test_label_noise_is_reproducible():
    assert _noisy([i % 10 for i in range(300)], list(range(300)), seed=5)[1] == \
        _noisy([i % 10 for i in range(300)], list(range(300)), seed=5)[1]


# -------------------------------------------------------- evaluation (L4)

def test_evaluate_averages_loss_per_sample():
    torch.manual_seed(0)
    model = torch.nn.Linear(3, 2)
    X, y = torch.randn(5, 3), torch.tensor([0, 1, 0, 1, 1])
    loader = DataLoader(TensorDataset(X, y), batch_size=4)      # batches of 4 and 1
    loss, _ = evaluate(model, loader, "cpu")
    expected = torch.nn.functional.cross_entropy(model(X), y).item()
    assert abs(loss - expected) < 1e-6
