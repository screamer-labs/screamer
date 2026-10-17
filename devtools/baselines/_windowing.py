"""Windowing helpers with no optional dependencies.

Kept apart from `_ohlc_vol`, which requires QuantLib: a reference that needs
only pandas must not be dragged out of an environment that lacks QuantLib.
"""
import numpy as np
import pandas as pd


def rolling_mean(values, window):
    """Trailing mean with NaN before the window fills, matching `strict`."""
    return pd.Series(np.asarray(values, dtype=float)).rolling(window).mean().to_numpy()


def ignoring_nan(values, compute):
    """compute() on the input with its NaN steps removed, with NaN put back at
    those steps: the output of an operator whose NaN policy is ``ignore``."""
    values = np.asarray(values, dtype=float)
    keep = ~np.isnan(values)
    part = np.asarray(compute(values[keep]), dtype=float)
    out = np.full((len(values),) + part.shape[1:], np.nan)
    out[keep] = part
    return out


def trailing(values, window, start_policy, minimum=1):
    """The trailing windows of ``values`` as a list (None where the policy
    gives no output): ``strict`` needs ``window`` values, ``expanding`` at
    least ``minimum``, and ``zero`` sees ``window`` zeros before the input."""
    values = np.asarray(values, dtype=float)
    pad = window if start_policy == "zero" else 0
    x = np.concatenate([np.zeros(pad), values])
    out = []
    for t in range(pad, len(x)):
        w = x[max(0, t - window + 1): t + 1]
        need = window if start_policy == "strict" else minimum
        out.append(w if len(w) >= need else None)
    return out
