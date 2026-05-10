"""
Tests for RollingArgmin / RollingArgmax.

Convention: 0 = oldest sample in the current window, window_size-1 =
newest. Matches numpy.argmin / numpy.argmax applied to the window
slice, and pandas.rolling().apply(np.argmin / np.argmax).

Algorithm uses the same monotonic-deque primitive as RollingMin /
RollingMax, exposed via the front element's window offset.
"""
import numpy as np
import pandas as pd
import pytest

from screamer import RollingArgmin, RollingArgmax, RollingMin, RollingMax


def _expanding_window_argextremum(x, w, fn):
    """Reference: numpy fn (argmin or argmax) over the trailing window."""
    out = np.empty(len(x))
    for i in range(len(x)):
        start = max(0, i - w + 1)
        out[i] = fn(x[start:i + 1])
    return out


@pytest.mark.parametrize("w", [3, 5, 10, 20])
def test_argmin_matches_numpy(w):
    rng = np.random.default_rng(w)
    x = rng.standard_normal(60)
    ref = _expanding_window_argextremum(x, w, np.argmin)
    np.testing.assert_array_equal(RollingArgmin(w)(x), ref)


@pytest.mark.parametrize("w", [3, 5, 10, 20])
def test_argmax_matches_numpy(w):
    rng = np.random.default_rng(w + 100)
    x = rng.standard_normal(60)
    ref = _expanding_window_argextremum(x, w, np.argmax)
    np.testing.assert_array_equal(RollingArgmax(w)(x), ref)


def test_argmin_matches_pandas():
    rng = np.random.default_rng(7)
    x = rng.standard_normal(50)
    w = 5
    ours = RollingArgmin(w)(x)
    ref = pd.Series(x).rolling(w).apply(np.argmin, raw=True).to_numpy()
    # Pandas returns NaN during warmup; we return the partial-window argmin.
    np.testing.assert_array_equal(ours[w - 1:], ref[w - 1:])


def test_argmax_matches_pandas():
    rng = np.random.default_rng(8)
    x = rng.standard_normal(50)
    w = 5
    ours = RollingArgmax(w)(x)
    ref = pd.Series(x).rolling(w).apply(np.argmax, raw=True).to_numpy()
    np.testing.assert_array_equal(ours[w - 1:], ref[w - 1:])


def test_argmin_consistent_with_rolling_min():
    """The value at the argmin index of the window must equal RollingMin."""
    rng = np.random.default_rng(9)
    x = rng.standard_normal(80)
    w = 6
    idx = RollingArgmin(w)(x).astype(int)
    val = RollingMin(w)(x)
    for i in range(len(x)):
        start = max(0, i - w + 1)
        assert x[start + idx[i]] == val[i]


def test_argmax_consistent_with_rolling_max():
    rng = np.random.default_rng(10)
    x = rng.standard_normal(80)
    w = 6
    idx = RollingArgmax(w)(x).astype(int)
    val = RollingMax(w)(x)
    for i in range(len(x)):
        start = max(0, i - w + 1)
        assert x[start + idx[i]] == val[i]


def test_warmup_offsets_are_in_range():
    """During warmup the offset must lie in [0, samples_seen-1]."""
    rng = np.random.default_rng(11)
    x = rng.standard_normal(20)
    w = 10
    out = RollingArgmin(w)(x)
    for i in range(len(x)):
        upper = min(i, w - 1)
        assert 0 <= out[i] <= upper


def test_argmin_argmax_for_strictly_monotonic_input():
    """Monotonically increasing -> argmin = 0 (oldest), argmax = w-1 (newest)."""
    x = np.arange(20, dtype=float)
    w = 5
    out_argmin = RollingArgmin(w)(x)
    out_argmax = RollingArgmax(w)(x)
    # After warmup, argmin is always 0 (oldest in window is smallest).
    np.testing.assert_array_equal(out_argmin[w - 1:], 0.0)
    # After warmup, argmax is always w-1 (newest in window is largest).
    np.testing.assert_array_equal(out_argmax[w - 1:], w - 1)


@pytest.mark.parametrize("cls", [RollingArgmin, RollingArgmax])
def test_scalar_loop_matches_array(cls):
    rng = np.random.default_rng(12)
    x = rng.standard_normal(40)
    w = 5
    obj = cls(w)
    streamed = np.array([obj(v) for v in x])
    np.testing.assert_array_equal(streamed, cls(w)(x))


@pytest.mark.parametrize("cls", [RollingArgmin, RollingArgmax])
def test_2d_per_column_independence(cls):
    rng = np.random.default_rng(13)
    X = rng.standard_normal((30, 4))
    w = 5
    out_2d = cls(w)(X)
    for k in range(X.shape[1]):
        np.testing.assert_array_equal(out_2d[:, k], cls(w)(X[:, k].copy()))
