"""Reference implementations for the benchmark suite.

For every benchmarked function there is a screamer variant and one or more
reference variants (numpy / pandas / scipy), named `<Func>__<lib>`. The runner
times each and the plot scripts compare them. `all()` returns the metadata table
(func, lib, callable, args) built by introspection.

These are speed references, not correctness checks. The runner feeds the same
`np.random.normal` array to every variant, so functions that expect positive
prices (Log, Sqrt, LogReturn) run on signed data; that is fine for timing and we
silence the resulting warnings.
"""
import inspect
import sys
import warnings

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view as swv
from scipy import signal as sp_signal
from scipy import special as sp_special

import screamer as sc

warnings.filterwarnings("ignore")

# argument string per function, as passed to the runner's eval()
_ARGS = {
    "Abs": "array", "Sign": "array", "Sqrt": "array", "Exp": "array", "Log": "array",
    "Erf": "array", "Erfc": "array",
    "Clip": "array,lower,upper", "Butter2": "array,cutoff_freq",
    "Ffill": "array", "FillNa": "array,fill",
    "Lag": "array,window_size", "Diff": "array,window_size",
    "Return": "array,window_size", "LogReturn": "array,window_size",
    "EwMean": "array,window_size", "EwStd": "array,window_size",
    "EwVar": "array,window_size", "EwZscore": "array,window_size",
    "RollingMean": "array,window_size", "RollingSum": "array,window_size",
    "RollingMax": "array,window_size", "RollingMin": "array,window_size",
    "RollingStd": "array,window_size", "RollingVar": "array,window_size",
    "RollingMedian": "array,window_size", "RollingRms": "array,window_size",
    "RollingZscore": "array,window_size", "RollingSkew": "array,window_size",
    "RollingKurt": "array,window_size",
    "RollingQuantile": "array,window_size,quantile",
    "RollingPoly1": "array,window_size,derivative_order",
}


# --- elementwise --------------------------------------------------------------
def Abs__screamer(array): return sc.Abs()(array)
def Abs__numpy(array): return np.abs(array)

def Sign__screamer(array): return sc.Sign()(array)
def Sign__numpy(array): return np.sign(array)

def Sqrt__screamer(array): return sc.Sqrt()(array)
def Sqrt__numpy(array): return np.sqrt(np.abs(array))

def Exp__screamer(array): return sc.Exp()(array)
def Exp__numpy(array): return np.exp(array)

def Log__screamer(array): return sc.Log()(array)
def Log__numpy(array): return np.log(np.abs(array))

def Erf__screamer(array): return sc.Erf()(array)
def Erf__scipy(array): return sp_special.erf(array)

def Erfc__screamer(array): return sc.Erfc()(array)
def Erfc__scipy(array): return sp_special.erfc(array)

def Clip__screamer(array, lower, upper): return sc.Clip(lower, upper)(array)
def Clip__numpy(array, lower, upper): return np.clip(array, lower, upper)
def Clip__pandas(array, lower, upper): return pd.Series(array).clip(lower, upper).to_numpy()

def Butter2__screamer(array, cutoff_freq): return sc.Butter(2, cutoff_freq)(array)
def Butter2__scipy(array, cutoff_freq):
    b, a = sp_signal.butter(2, cutoff_freq)
    return sp_signal.lfilter(b, a, array)


# --- missing data / shifts ----------------------------------------------------
def Ffill__screamer(array): return sc.Ffill()(array)
def Ffill__pandas(array): return pd.Series(array).ffill().to_numpy()
def Ffill__numpy(array):
    mask = ~np.isnan(array)
    idx = np.where(mask, np.arange(len(array)), 0)
    np.maximum.accumulate(idx, out=idx)
    return array[idx]

def FillNa__screamer(array, fill): return sc.FillNa(fill)(array)
def FillNa__numpy(array, fill): return np.where(np.isnan(array), fill, array)
def FillNa__pandas(array, fill): return pd.Series(array).fillna(fill).to_numpy()

def Lag__screamer(array, window_size): return sc.Lag(window_size)(array)
def Lag__numpy(array, window_size):
    out = np.empty_like(array)
    out[:window_size] = np.nan
    out[window_size:] = array[:-window_size]
    return out
def Lag__pandas(array, window_size): return pd.Series(array).shift(window_size).to_numpy()

def Diff__screamer(array, window_size): return sc.Diff(window_size)(array)
def Diff__numpy(array, window_size): return array[window_size:] - array[:-window_size]
def Diff__pandas(array, window_size): return pd.Series(array).diff(window_size).to_numpy()

def Return__screamer(array, window_size): return sc.Return(window_size)(array)
def Return__numpy(array, window_size): return array[window_size:] / array[:-window_size] - 1.0
def Return__pandas(array, window_size): return pd.Series(array).pct_change(window_size).to_numpy()

def LogReturn__screamer(array, window_size): return sc.LogReturn(window_size)(array)
def LogReturn__numpy(array, window_size):
    return np.log(np.abs(array[window_size:] / array[:-window_size]))
def LogReturn__pandas(array, window_size):
    return np.log(pd.Series(array).abs()).diff(window_size).to_numpy()


# --- exponentially weighted (span = window_size) ------------------------------
def _alpha(w): return 2.0 / (w + 1.0)

def EwMean__screamer(array, window_size): return sc.EwMean(span=window_size)(array)
def EwMean__pandas(array, window_size): return pd.Series(array).ewm(span=window_size).mean().to_numpy()
def EwMean__numpy(array, window_size):
    a = _alpha(window_size)
    return sp_signal.lfilter([a], [1.0, -(1.0 - a)], array)

def EwStd__screamer(array, window_size): return sc.EwStd(span=window_size)(array)
def EwStd__pandas(array, window_size): return pd.Series(array).ewm(span=window_size).std().to_numpy()

def EwVar__screamer(array, window_size): return sc.EwVar(span=window_size)(array)
def EwVar__pandas(array, window_size): return pd.Series(array).ewm(span=window_size).var().to_numpy()

def EwZscore__screamer(array, window_size): return sc.EwZscore(span=window_size)(array)
def EwZscore__pandas(array, window_size):
    s = pd.Series(array)
    ew = s.ewm(span=window_size)
    return ((s - ew.mean()) / ew.std()).to_numpy()


# --- rolling ------------------------------------------------------------------
def RollingMean__screamer(array, window_size): return sc.RollingMean(window_size)(array)
def RollingMean__pandas(array, window_size): return pd.Series(array).rolling(window_size).mean().to_numpy()
def RollingMean__numpy(array, window_size): return swv(array, window_size).mean(axis=1)

def RollingSum__screamer(array, window_size): return sc.RollingSum(window_size)(array)
def RollingSum__pandas(array, window_size): return pd.Series(array).rolling(window_size).sum().to_numpy()
def RollingSum__numpy(array, window_size): return swv(array, window_size).sum(axis=1)

def RollingMax__screamer(array, window_size): return sc.RollingMax(window_size)(array)
def RollingMax__pandas(array, window_size): return pd.Series(array).rolling(window_size).max().to_numpy()
def RollingMax__numpy(array, window_size): return swv(array, window_size).max(axis=1)

def RollingMin__screamer(array, window_size): return sc.RollingMin(window_size)(array)
def RollingMin__pandas(array, window_size): return pd.Series(array).rolling(window_size).min().to_numpy()
def RollingMin__numpy(array, window_size): return swv(array, window_size).min(axis=1)

def RollingStd__screamer(array, window_size): return sc.RollingStd(window_size)(array)
def RollingStd__pandas(array, window_size): return pd.Series(array).rolling(window_size).std().to_numpy()
def RollingStd__numpy(array, window_size): return swv(array, window_size).std(axis=1, ddof=1)

def RollingVar__screamer(array, window_size): return sc.RollingVar(window_size)(array)
def RollingVar__pandas(array, window_size): return pd.Series(array).rolling(window_size).var().to_numpy()
def RollingVar__numpy(array, window_size): return swv(array, window_size).var(axis=1, ddof=1)

def RollingMedian__screamer(array, window_size): return sc.RollingMedian(window_size)(array)
def RollingMedian__pandas(array, window_size): return pd.Series(array).rolling(window_size).median().to_numpy()
def RollingMedian__numpy(array, window_size): return np.median(swv(array, window_size), axis=1)

def RollingRms__screamer(array, window_size): return sc.RollingRms(window_size)(array)
def RollingRms__pandas(array, window_size):
    return np.sqrt((pd.Series(array) ** 2).rolling(window_size).mean()).to_numpy()
def RollingRms__numpy(array, window_size): return np.sqrt((swv(array, window_size) ** 2).mean(axis=1))

def RollingZscore__screamer(array, window_size): return sc.RollingZscore(window_size)(array)
def RollingZscore__pandas(array, window_size):
    s = pd.Series(array); r = s.rolling(window_size)
    return ((s - r.mean()) / r.std()).to_numpy()
def RollingZscore__numpy(array, window_size):
    w = swv(array, window_size)
    return (array[window_size - 1:] - w.mean(axis=1)) / w.std(axis=1, ddof=1)

def RollingSkew__screamer(array, window_size): return sc.RollingSkew(window_size)(array)
def RollingSkew__pandas(array, window_size): return pd.Series(array).rolling(window_size).skew().to_numpy()
def RollingSkew__numpy(array, window_size):
    w = swv(array, window_size)
    d = w - w.mean(axis=1, keepdims=True)
    return (d ** 3).mean(axis=1) / (d ** 2).mean(axis=1) ** 1.5

def RollingKurt__screamer(array, window_size): return sc.RollingKurt(window_size)(array)
def RollingKurt__pandas(array, window_size): return pd.Series(array).rolling(window_size).kurt().to_numpy()
def RollingKurt__numpy(array, window_size):
    w = swv(array, window_size)
    d = w - w.mean(axis=1, keepdims=True)
    return (d ** 4).mean(axis=1) / (d ** 2).mean(axis=1) ** 2 - 3.0

def RollingQuantile__screamer(array, window_size, quantile):
    return sc.RollingQuantile(window_size, quantile)(array)
def RollingQuantile__pandas(array, window_size, quantile):
    return pd.Series(array).rolling(window_size).quantile(quantile).to_numpy()
def RollingQuantile__numpy(array, window_size, quantile):
    return np.quantile(swv(array, window_size), quantile, axis=1)

def RollingPoly1__screamer(array, window_size, derivative_order):
    return sc.RollingPoly1(window_size, derivative_order)(array)
def _poly1(w_data, derivative_order):
    x = np.arange(w_data.shape[1]); xm = x.mean()
    slope = (w_data * (x - xm)).sum(axis=1) / ((x - xm) ** 2).sum()
    if derivative_order == 1:
        return slope
    return w_data.mean(axis=1) + slope * ((w_data.shape[1] - 1) - xm)
def RollingPoly1__numpy(array, window_size, derivative_order):
    return _poly1(swv(array, window_size), derivative_order)
def RollingPoly1__pandas(array, window_size, derivative_order):
    return _poly1(swv(pd.Series(array).to_numpy(), window_size), derivative_order)


def all():
    """Metadata table of every `<Func>__<lib>` variant defined in this module."""
    rows = []
    for name, obj in inspect.getmembers(sys.modules[__name__], inspect.isfunction):
        if name.startswith("_") or "__" not in name:
            continue
        func, lib = name.split("__")
        rows.append(dict(func=func, lib=lib, callable=name, var="", args=_ARGS[func]))
    return pd.DataFrame(rows).sort_values(["func", "lib"]).reset_index(drop=True)
