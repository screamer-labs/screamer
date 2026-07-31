"""Windowing helpers with no optional dependencies.

Kept apart from `_ohlc_vol`, which requires QuantLib: a reference that needs
only pandas must not be dragged out of an environment that lacks QuantLib.
"""
import numpy as np
import pandas as pd


def rolling_mean(values, window):
    """Trailing mean with NaN before the window fills, matching `strict`."""
    return pd.Series(np.asarray(values, dtype=float)).rolling(window).mean().to_numpy()
