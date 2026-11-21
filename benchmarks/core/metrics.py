"""Metrics computed from a run's per-epoch history (pure functions)."""
from typing import Dict, List, Optional, Sequence


def post_shift_metrics(val_losses: Sequence[float], shift_epoch: Optional[int],
                       epsilon: float = 0.05) -> Dict[str, Optional[float]]:
    """
    How a run coped with a shift at `shift_epoch` (the first epoch trained under it).

    baseline:     last pre-shift validation loss (frozen at the shift)
    degradation:  mean post-shift loss minus the baseline
    loss_spike:   largest epoch-to-epoch loss increase after the shift
    recovery:     epochs from the shift until loss <= (1 + epsilon) * best
                  pre-shift loss (0 = never degraded; None = never recovered)
    """
    if shift_epoch is None or shift_epoch <= 0 or shift_epoch >= len(val_losses):
        return {"degradation": None, "loss_spike": None, "recovery_epochs": None}
    pre, post = list(val_losses[:shift_epoch]), list(val_losses[shift_epoch:])
    baseline = pre[-1]
    threshold = min(pre) * (1 + epsilon)
    recovery = next((t for t, loss in enumerate(post) if loss <= threshold), None)
    spikes = [val_losses[i] - val_losses[i - 1] for i in range(shift_epoch, len(val_losses))]
    return {
        "degradation": sum(post) / len(post) - baseline,
        "loss_spike": max(spikes),
        "recovery_epochs": recovery,
    }


def cumulative_regret(controller_val_losses: Sequence[float], baseline_val_losses: Sequence[float]) -> float:
    """Sum over epochs of (controller - baseline) validation loss; negative is better."""
    return float(sum(c - b for c, b in zip(controller_val_losses, baseline_val_losses)))


def val_losses(history: List[dict]) -> List[float]:
    return [h["val_loss"] for h in history]
