import time

class ComputeMetrics:
    def __init__(self):
        self.num_hparam_updates = 0
        self.num_trials_started = 0
        self.num_rollbacks = 0

        self.controller_time_sec = 0.0
        self.total_epochs = 0

        self._current_trial_epochs = 0
        self._trial_lengths = []

    def start_trial(self):
        self.num_trials_started += 1
        self._current_trial_epochs = 0

    def step_epoch(self):
        self._current_trial_epochs += 1
        self.total_epochs += 1

    def end_trial(self):
        if self._current_trial_epochs > 0:
            self._trial_lengths.append(self._current_trial_epochs)

    def record_update(self):
        self.num_hparam_updates += 1

    def record_rollback(self):
        self.num_rollbacks += 1

    def record_controller_time(self, dt: float):
        self.controller_time_sec += dt

    def summary(self):
        avg_epochs = (
            sum(self._trial_lengths) / len(self._trial_lengths)
            if self._trial_lengths else 0
        )

        return {
            "num_hparam_updates": self.num_hparam_updates,
            "num_trials_started": self.num_trials_started,
            "num_rollbacks": self.num_rollbacks,
            "epochs_per_trial_avg": avg_epochs,
            "controller_time_sec": round(self.controller_time_sec, 4),
            "controller_time_per_epoch_ms": round(
                (self.controller_time_sec / max(self.total_epochs, 1)) * 1000, 4
            ),
        }
