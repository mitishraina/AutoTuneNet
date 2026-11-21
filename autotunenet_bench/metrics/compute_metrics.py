class ComputeMetrics:
    """
    Controller cost over one run.

    A trial is the stretch of epochs one proposed configuration runs for: it
    starts when the controller applies a new proposal and ends at the next
    proposal, a rollback, or the end of the run. An update is an epoch at
    whose end the tuned values changed (a proposal or a rollback).
    """

    def __init__(self):
        self.num_hparam_updates = 0
        self.num_trials_started = 0
        self.num_rollbacks = 0

        self.controller_time_sec = 0.0
        self.total_epochs = 0

        self._trial_open = False
        self._current_trial_epochs = 0
        self._trial_lengths = []

    def start_trial(self):
        self.end_trial()
        self.num_trials_started += 1
        self._trial_open = True
        self._current_trial_epochs = 0

    def step_epoch(self):
        """Count an epoch that has finished training."""
        self.total_epochs += 1
        if self._trial_open:
            self._current_trial_epochs += 1

    def end_trial(self):
        if self._trial_open and self._current_trial_epochs > 0:
            self._trial_lengths.append(self._current_trial_epochs)
        self._trial_open = False
        self._current_trial_epochs = 0

    def record_update(self):
        self.num_hparam_updates += 1

    def record_rollback(self):
        self.num_rollbacks += 1

    def record_controller_time(self, dt: float):
        self.controller_time_sec += dt

    def summary(self):
        lengths = list(self._trial_lengths)
        if self._trial_open and self._current_trial_epochs > 0:
            lengths.append(self._current_trial_epochs)   # trial still running at the end
        avg_epochs = sum(lengths) / len(lengths) if lengths else 0

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
