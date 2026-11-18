import torch
from torch.utils.data import DataLoader, TensorDataset

from autotunenet_bench.regimes.factory import build_regime
from autotunenet_bench.regimes.optimizer_noise import GradientNoiseRegime
from benchmarks.common.train_utils import train_epoch


def _train(regime=None, epochs=3):
    """Train a small seeded model, calling the regime the way the runner does."""
    torch.manual_seed(0)
    X = torch.randn(64, 4)
    y = torch.randint(0, 3, (64,))
    loader = DataLoader(TensorDataset(X, y), batch_size=16, shuffle=False)
    model = torch.nn.Linear(4, 3)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    snapshots = []
    for epoch in range(epochs):
        if regime is not None:
            regime.maybe_apply(epoch=epoch, model=model, optimizer=optimizer, train_dataset=None)
        train_epoch(model, loader, optimizer, "cpu")
        snapshots.append(torch.cat([p.detach().flatten().clone() for p in model.parameters()]))
    return snapshots


def test_noise_changes_training_after_the_shift():
    clean = _train()
    noisy = _train(GradientNoiseRegime(start_epoch=1, std=0.5))
    assert not torch.allclose(clean[-1], noisy[-1])


def test_no_effect_before_the_shift_epoch():
    clean = _train()
    noisy = _train(GradientNoiseRegime(start_epoch=2, std=0.5))
    # Epochs 0 and 1 run before the regime activates.
    assert torch.equal(clean[0], noisy[0])
    assert torch.equal(clean[1], noisy[1])
    assert not torch.equal(clean[2], noisy[2])


def test_noise_is_reproducible_for_a_fixed_seed():
    a = _train(GradientNoiseRegime(start_epoch=1, std=0.5, seed=7))
    b = _train(GradientNoiseRegime(start_epoch=1, std=0.5, seed=7))
    c = _train(GradientNoiseRegime(start_epoch=1, std=0.5, seed=8))
    assert torch.equal(a[-1], b[-1])
    assert not torch.equal(a[-1], c[-1])


def test_noise_does_not_touch_the_global_rng():
    torch.manual_seed(123)
    expected = torch.rand(3)

    torch.manual_seed(123)
    _ = GradientNoiseRegime(start_epoch=0, std=0.5)._add_noise(torch.zeros(10))
    assert torch.equal(torch.rand(3), expected)


def test_remove_detaches_the_hooks():
    regime = GradientNoiseRegime(start_epoch=0, std=0.5)
    model = torch.nn.Linear(2, 1)
    regime.maybe_apply(epoch=0, model=model)
    regime.remove()

    model(torch.ones(1, 2)).sum().backward()
    assert torch.equal(model.weight.grad, torch.ones(1, 2))


def test_factory_passes_std_and_seed():
    regime = build_regime({"type": "gradient_noise", "start_epoch": 5, "std": 0.02, "seed": 3})
    assert (regime.start_epoch, regime.std, regime.seed) == (5, 0.02, 3)
