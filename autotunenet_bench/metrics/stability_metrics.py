from dataclasses import dataclass, field
from typing import List, Optional
from autotunenet_bench.metrics.recovery import TimeToRecovery

@dataclass
class StabilityMetrics:
    """
    Tracks stability related metrics during a single training run
    """
    val_losses: List[float] = field(default_factory=list)
    instability_epochs: List[int] = field(default_factory=list)
    rollback_epochs: List[int] = field(default_factory=list)
    
    pre_shift_best: Optional[float] = None
    shift_epoch: Optional[int] = None
    
    recovered_epoch: Optional[int] = None
    recovery_epsilion: float = 0.05
    
    reset_best_on_shift: bool = True
    
    def record_loss(self, loss: float):
        self.val_losses.append(loss)
        
        if self.pre_shift_best is None:
            self.pre_shift_best = loss
        else:
            self.pre_shift_best = min(self.pre_shift_best, loss)
            
    def mark_instability(self, epoch: int):
        if epoch not in self.instability_epochs:
            self.instability_epochs.append(epoch)
            
    def mark_rollback(self, epoch: int):
        if epoch not in self.rollback_epochs:
            self.rollback_epochs.append(epoch)
            
    def mark_shift(self, epoch: int):
        self.shift_epoch = epoch
        
        if self.reset_best_on_shift:
            if self.val_losses:
                self.pre_shift_best = self.val_losses[-1]
        
    def check_recovery(self, epoch: int, loss: float):
        if self.shift_epoch is None or self.recovered_epoch is not None:
            return
        
        if loss <= (1 + self.recovery_epsilion) * self.pre_shift_best:
            self.recovered_epoch = epoch
            
    def summary(self) -> dict:
        post_shift_losses = (
            self.val_losses[self.shift_epoch + 1:]
            if self.shift_epoch is not None
            else []
        )
        
        result = {
            "instability_events": len(self.instability_epochs),
            "rollback_count": len(self.rollback_epochs),
            "time_to_first_instability": (
                self.instability_epochs[0]
                if self.instability_epochs
                else None
            ),
            "recovery_time": (
                None
                if self.recovered_epoch is None or self.shift_epoch is None
                else self.recovered_epoch - self.shift_epoch
            ),
            "post_shift_degradation": (
                None
                if not post_shift_losses
                else sum(post_shift_losses) / len(post_shift_losses)
                - self.pre_shift_best
            ),
            "loss_spike_magnitude": self._max_spike(),
            "shift_epoch": self.shift_epoch
        }
        
        if self.shift_epoch is not None:
            ttr = TimeToRecovery(epsilon=0.05).compute(
                losses=self.val_losses,
                shift_epoch=self.shift_epoch,
            )
            result["time_to_recovery_epochs"] = (
                ttr if ttr != float("inf") else None
            )

        return result
        
    def _max_spike(self) -> float:
        if len(self.val_losses) < 2:
            return 0.0
        return max(
            self.val_losses[i] - self.val_losses[i - 1]
            for i in range(1, len(self.val_losses))
        )