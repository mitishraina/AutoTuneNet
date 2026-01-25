# TTR: Time to Recover Metric

class TimeToRecovery:
    def __init__(self, epsilon: float = 0.05):
        self.epsilon = epsilon

    def compute(self, losses, shift_epoch):
        pre_shift_best = min(losses[:shift_epoch])

        threshold = pre_shift_best * (1 + self.epsilon)

        for t in range(shift_epoch + 1, len(losses)):
            if losses[t] <= threshold:
                return t - shift_epoch

        return float("inf")
