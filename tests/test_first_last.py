"""Tests for First and Last reducer functors.

First: latches the first finite value seen since reset().
Last:  tracks the most recent finite value since reset().

Both follow NaN policy "ignore": NaN in -> NaN out, state unchanged.
"""
import numpy as np
import pytest

from screamer import First, Last
from screamer.streams import resample, Resample


# ---------------------------------------------------------------------------
# Basic semantics
# ---------------------------------------------------------------------------

def test_first_latches_first_value():
    x = np.array([3.0, 1.0, 4.0, 1.0, 5.0])
    out = First()(x)
    np.testing.assert_array_equal(out, [3.0, 3.0, 3.0, 3.0, 3.0])


def test_last_tracks_latest_value():
    x = np.array([3.0, 1.0, 4.0, 1.0, 5.0])
    out = Last()(x)
    np.testing.assert_array_equal(out, [3.0, 1.0, 4.0, 1.0, 5.0])


# ---------------------------------------------------------------------------
# NaN handling
# ---------------------------------------------------------------------------

def test_first_nan_handling():
    """NaN in -> NaN out; the latched first value is unchanged."""
    x = np.array([np.nan, 2.0, np.nan, 3.0])
    out = First()(x)
    assert np.isnan(out[0])           # leading NaN before any finite value
    assert out[1] == 2.0              # first finite value latched
    assert np.isnan(out[2])           # NaN in -> NaN out, latch unchanged
    assert out[3] == 2.0              # latch is still 2.0


def test_last_nan_handling():
    """NaN in -> NaN out; the retained last value is unchanged."""
    x = np.array([np.nan, 2.0, np.nan, 3.0])
    out = Last()(x)
    assert np.isnan(out[0])           # no finite value seen yet
    assert out[1] == 2.0              # first finite value
    assert np.isnan(out[2])           # NaN in -> NaN out, last unchanged
    assert out[3] == 3.0              # new finite value retained


# ---------------------------------------------------------------------------
# reset()
# ---------------------------------------------------------------------------

def test_first_reset_restarts_latch():
    f = First()
    x1 = np.array([10.0, 20.0, 30.0])
    out1 = f(x1)
    assert out1[-1] == 10.0           # latched 10

    f.reset()
    x2 = np.array([99.0, 50.0])
    out2 = f(x2)
    assert out2[0] == 99.0            # latch reset; 99 is the new first


def test_last_reset_restarts_track():
    la = Last()
    x1 = np.array([5.0, 6.0, 7.0])
    la(x1)
    la.reset()
    x2 = np.array([1.0, 2.0])
    out2 = la(x2)
    np.testing.assert_array_equal(out2, [1.0, 2.0])


# ---------------------------------------------------------------------------
# Resample-reducer equivalence (the key anchor)
# ---------------------------------------------------------------------------

def _make_resample_data(bar_width=5, n_bars=4):
    """Monotone index, simple values, no NaNs - clean bars of width bar_width."""
    n = bar_width * n_bars
    idx = np.arange(n, dtype=np.int64)
    vals = np.arange(1.0, n + 1.0)
    return vals, idx, bar_width


def test_first_functor_equals_builtin_first():
    vals, idx, w = _make_resample_data()
    v_str, k_str = Resample(freq=w, agg="first")(vals, idx)
    v_fun, k_fun = Resample(freq=w, agg=First())(vals, idx)
    np.testing.assert_array_equal(np.asarray(k_str), np.asarray(k_fun))
    np.testing.assert_array_equal(np.asarray(v_str), np.asarray(v_fun))


def test_last_functor_equals_builtin_last():
    vals, idx, w = _make_resample_data()
    v_str, k_str = Resample(freq=w, agg="last")(vals, idx)
    v_fun, k_fun = Resample(freq=w, agg=Last())(vals, idx)
    np.testing.assert_array_equal(np.asarray(k_str), np.asarray(k_fun))
    np.testing.assert_array_equal(np.asarray(v_str), np.asarray(v_fun))
