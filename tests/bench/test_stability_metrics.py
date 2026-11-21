import pytest

from autotunenet_bench.metrics.stability_metrics import StabilityMetrics
from autotunenet_bench.metrics.recovery import TimeToRecovery


def _run(losses, shift_epoch, **kwargs):
    m = StabilityMetrics(**kwargs)
    for epoch, loss in enumerate(losses):
        if epoch == shift_epoch:
            m.mark_shift(epoch)      # the runner marks the shift before recording
        m.record_loss(loss)
    return m


def test_baseline_is_frozen_at_the_shift():
    m = _run([0.5, 0.4, 0.3, 0.6, 0.2], shift_epoch=3)
    assert m.pre_shift_best == 0.3          # not the post-shift 0.2


def test_degradation_uses_the_shift_epoch_and_a_pre_shift_baseline():
    m = _run([0.5, 0.4, 0.3, 0.6, 0.2], shift_epoch=3)
    s = m.summary()
    # post-shift window = [0.6, 0.2] -> mean 0.4; baseline 0.3
    assert s["post_shift_degradation"] == pytest.approx(0.1)


def test_reset_best_on_shift_chooses_the_baseline():
    losses = [0.5, 0.3, 0.4, 0.6, 0.35]
    assert _run(losses, 3, reset_best_on_shift=True).pre_shift_best == 0.4   # last pre-shift
    assert _run(losses, 3, reset_best_on_shift=False).pre_shift_best == 0.3  # best pre-shift


def test_recovery_time_counts_from_the_shift_epoch():
    m = _run([0.5, 0.4, 0.3, 0.6, 0.2], shift_epoch=3)
    for epoch, loss in enumerate(m.val_losses):
        m.check_recovery(epoch, loss)
    s = m.summary()
    assert s["recovery_time"] == 1          # recovered at epoch 4
    assert s["time_to_recovery_epochs"] == 1


def test_time_to_recovery_is_zero_when_the_shift_does_no_harm():
    assert TimeToRecovery(epsilon=0.05).compute([0.5, 0.4, 0.3, 0.3, 0.29], shift_epoch=3) == 0
