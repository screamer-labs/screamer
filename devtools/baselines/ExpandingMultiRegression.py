"""Least squares of y on k regressors over every row so far, by
numpy.linalg.lstsq (devtools/baselines/RollingMultiRegression.py with no window)."""
from devtools.baselines.RollingMultiRegression import ols_rows


class ExpandingMultiRegression_numpy:

    def __init__(self, n_regressors=3, intercept=True):
        self.intercept = intercept

    def __call__(self, y, *xs):
        return ols_rows(y, xs, None, self.intercept)
