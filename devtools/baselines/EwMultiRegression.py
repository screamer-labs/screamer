"""Exponentially weighted least squares of y on k regressors over every row
so far, by numpy.linalg.lstsq on rows scaled by the square root of their
weight (devtools/baselines/RollingMultiRegression.py with no window and
alpha)."""
import numpy as np

from devtools.baselines.RollingMultiRegression import ols_rows


class EwMultiRegression_numpy:

    def __init__(self, n_regressors=3, intercept=True, com=None, span=None, halflife=None, alpha=None):
        if alpha is None:
            if com is not None:
                alpha = 1.0 / (1.0 + com)
            elif span is not None:
                alpha = 2.0 / (span + 1.0)
            else:
                alpha = 1.0 - np.exp(-np.log(2.0) / halflife)
        self.intercept = intercept
        self.alpha = alpha

    def __call__(self, y, *xs):
        return ols_rows(y, xs, None, self.intercept, self.alpha)
