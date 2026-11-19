# TTR: Time to Recover Metric

class TimeToRecovery:
    def __init__(self, epsilon: float = 0.05):
        self.epsilon = epsilon

    def compute(self, losses, shift_epoch):
        """
        Epochs from the shift until the loss is back within (1 + epsilon) of the
        best pre-shift loss. losses[shift_epoch] is the first post-shift epoch,
        so 0 means it never degraded beyond the tolerance.
        """
        pre_shift_best = min(losses[:shift_epoch])

        threshold = pre_shift_best * (1 + self.epsilon)

        for t in range(shift_epoch, len(losses)):
            if losses[t] <= threshold:
                return t - shift_epoch

        return float("inf")
