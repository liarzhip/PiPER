import numpy as np

class EMAFilter:
    def __init__(self, alpha=0.25):
        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha must be in (0, 1]")
        self.alpha = float(alpha)
        self.value = None

    def reset(self):
        self.value = None

    def update(self, x):
        x = np.asarray(x, dtype=np.float64)
        self.value = x.copy() if self.value is None else (
            self.alpha * x + (1.0 - self.alpha) * self.value
        )
        return self.value.copy()
