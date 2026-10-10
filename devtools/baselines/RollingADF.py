"""Augmented Dickey-Fuller t-statistic over a rolling window, by least squares.

Each window holds `window_size` rows of dy_t on [1, y_{t-1}, dy_{t-1}, ...,
dy_{t-lags}]; the statistic is the t-value of the y_{t-1} coefficient.
"""
import numpy as np


class RollingADF_numpy:

    def __init__(self, window_size=100, lags=1):
        self.window_size, self.lags = window_size, lags

    def __call__(self, y):
        y = np.asarray(y, dtype=float)
        w, p = self.window_size, self.lags
        dy = np.diff(y)                       # dy[j - 1] = y[j] - y[j - 1]
        out = np.full(len(y), np.nan)
        for t in range(w + p, len(y)):
            js = np.arange(t - w + 1, t + 1)
            X = np.column_stack([np.ones(w), y[js - 1]] + [dy[js - 1 - i] for i in range(1, p + 1)])
            Y = dy[js - 1]
            if np.isnan(X).any() or np.isnan(Y).any():
                continue
            beta, *_ = np.linalg.lstsq(X, Y, rcond=None)
            r = Y - X @ beta
            s2 = r @ r / (w - X.shape[1])
            out[t] = beta[1] / np.sqrt(s2 * np.linalg.inv(X.T @ X)[1, 1])
        return out
