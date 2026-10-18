"""Least squares of y on k regressors over a trailing window, by
numpy.linalg.lstsq on each window. pred and resid use the fit of the window
that ends one row earlier; r2 is centered with an intercept and uncentered
without one. NaN rows are dropped before the windows are formed. With alpha,
the rows so far are weighted, the row i rows back by (1 - alpha)^i, and the
fit is the weighted one (lstsq on rows scaled by sqrt(weight))."""
import numpy as np


def ols_rows(y, xs, window, intercept, alpha=None):
    """(T, k + 3 [+ 1]) outputs; window None fits every row so far."""
    y = np.asarray(y, dtype=float)
    X = np.column_stack([np.asarray(x, dtype=float) for x in xs])
    k = X.shape[1]
    p = k + (1 if intercept else 0)
    need = window if window is not None else p + 1
    out = np.full((len(y), 3 + p), np.nan)
    keep = ~(np.isnan(y) | np.isnan(X).any(axis=1))
    rows = np.flatnonzero(keep)
    prev = None
    for i, t in enumerate(rows):
        lo = 0 if window is None else max(0, i + 1 - window)
        sel = rows[lo:i + 1]
        A = X[sel]
        if intercept:
            A = np.column_stack([np.ones(len(sel)), A])
        v = y[sel]
        w = np.ones(len(sel)) if alpha is None else (1.0 - alpha) ** np.arange(len(sel) - 1, -1, -1)
        sw = np.sqrt(w)
        b = None
        if len(sel) >= need and np.linalg.matrix_rank(A * sw[:, None]) == p:
            b = np.linalg.lstsq(A * sw[:, None], v * sw, rcond=None)[0]
        row = np.r_[1.0, X[t]] if intercept else X[t]
        if prev is not None:
            out[t, 0] = row @ prev
            out[t, 1] = y[t] - out[t, 0]
        if b is not None:
            e = v - A @ b
            ref = v - (w @ v) / w.sum() if intercept else v
            out[t, 2] = 1.0 - (w @ (e * e)) / (w @ (ref * ref))
            out[t, 3:] = b
        prev = b
    return out


class RollingMultiRegression_numpy:

    def __init__(self, window_size=250, n_regressors=3, intercept=True):
        self.window_size = window_size
        self.intercept = intercept

    def __call__(self, y, *xs):
        return ols_rows(y, xs, self.window_size, self.intercept)
