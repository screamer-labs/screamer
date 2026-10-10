"""Total least squares slope of x on y over a rolling window.

The slope of the first principal component of the centered (y, x) points:
the eigenvector of their 2 x 2 covariance matrix with the larger eigenvalue.
"""
import numpy as np


class RollingTLSBeta_numpy:

    def __init__(self, window_size=20, start_policy="strict"):
        self.window_size = window_size

    def __call__(self, x, y):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        w = self.window_size
        out = np.full(len(x), np.nan)
        for t in range(w - 1, len(x)):
            pts = np.column_stack([y[t - w + 1:t + 1], x[t - w + 1:t + 1]])
            if np.isnan(pts).any():
                continue
            vals, vecs = np.linalg.eigh(np.cov(pts, rowvar=False))
            v = vecs[:, np.argmax(vals)]
            out[t] = v[1] / v[0] if v[0] != 0 else np.nan
        return out
