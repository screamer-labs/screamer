"""Two-input affine combination, a*x + b*y + c."""
import numpy as np


class Linear2_numpy:

    def __init__(self, a=1.0, b=1.0, c=0.0):
        self.a, self.b, self.c = a, b, c

    def __call__(self, x, y):
        return self.a * np.asarray(x, dtype=float) + self.b * np.asarray(y, dtype=float) + self.c
