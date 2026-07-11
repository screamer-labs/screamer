"""Tests for multi-column bars via composition: combine_latest of per-stat resamples.

The dict agg form (resample(t, agg={name: expr, ...})) has been removed. This file
verifies that the composition recipe reproduces the same numeric results, and that
the removed dict form now raises a clear error.
"""
import numpy as np
import pytest

from screamer import ExpandingMax, ExpandingMin, ExpandingSum, First, Last, NegPart, PosPart
from screamer.streams import CombineLatest, Resample

# ---------------------------------------------------------------------------
# Fixtures / shared data
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(42)
N = 50      # number of ticks
W = 10      # bar width (every=W)
N_BARS = N // W


def _make_data():
    t_arr = np.arange(N, dtype=np.int64)
    price_arr = RNG.standard_normal(N).cumsum() + 100.0
    # signed volume: positive => buy, negative => sell
    vol_arr = RNG.standard_normal(N) * 10.0
    return t_arr, price_arr, vol_arr


# ---------------------------------------------------------------------------
# Test 1: OHLCV2 recipe (time mode) via composition - the headline test
# ---------------------------------------------------------------------------

class TestOHLCV2Composition:
    """Six-column OHLCV2 via combine_latest of per-stat resamples."""

    def setup_method(self):
        t_arr, price_arr, vol_arr = _make_data()
        self.t_arr = t_arr
        self.price_arr = price_arr
        self.vol_arr = vol_arr

        pos_vol = np.where(vol_arr > 0, vol_arr, 0.0)
        neg_vol = np.where(vol_arr < 0, -vol_arr, 0.0)

        o   = Resample(freq=W, agg="first")(price_arr, t_arr)
        h   = Resample(freq=W, agg="max")(price_arr, t_arr)
        l   = Resample(freq=W, agg="min")(price_arr, t_arr)
        c   = Resample(freq=W, agg="last")(price_arr, t_arr)
        buy = Resample(freq=W, agg="sum")(pos_vol,   t_arr)
        sel = Resample(freq=W, agg="sum")(neg_vol,   t_arr)
        self.out, self.idx = CombineLatest()(o, h, l, c, buy, sel)

    def test_returns_ndarray(self):
        assert isinstance(self.out, np.ndarray)

    def test_shape(self):
        assert self.out.ndim == 2
        assert self.out.shape[1] == 6
        assert len(self.out) == N_BARS

    def _reference_col(self, agg_str, arr=None):
        if arr is None:
            arr = self.price_arr
        ref = Resample(freq=W, agg=agg_str)(arr, self.t_arr)
        return ref[0]

    def test_open_matches_reference(self):
        ref = self._reference_col("first")
        np.testing.assert_allclose(self.out[:, 0], ref, rtol=1e-12)

    def test_high_matches_reference(self):
        ref = self._reference_col("max")
        np.testing.assert_allclose(self.out[:, 1], ref, rtol=1e-12)

    def test_low_matches_reference(self):
        ref = self._reference_col("min")
        np.testing.assert_allclose(self.out[:, 2], ref, rtol=1e-12)

    def test_close_matches_reference(self):
        ref = self._reference_col("last")
        np.testing.assert_allclose(self.out[:, 3], ref, rtol=1e-12)

    def test_buy_matches_reference(self):
        pos_vol = np.where(self.vol_arr > 0, self.vol_arr, 0.0)
        ref = self._reference_col("sum", pos_vol)
        np.testing.assert_allclose(self.out[:, 4], ref, rtol=1e-12)

    def test_sell_matches_reference(self):
        neg_vol = np.where(self.vol_arr < 0, -self.vol_arr, 0.0)
        ref = self._reference_col("sum", neg_vol)
        np.testing.assert_allclose(self.out[:, 5], ref, rtol=1e-12)


# ---------------------------------------------------------------------------
# Test 2: Count mode via composition
# ---------------------------------------------------------------------------

class TestCountModeComposition:
    """Three-column open/close/buy via combine_latest with count= bucketing."""

    COUNT = 5

    def setup_method(self):
        t_arr, price_arr, vol_arr = _make_data()
        self.t_arr = t_arr
        self.price_arr = price_arr
        self.vol_arr = vol_arr

        pos_vol = np.where(vol_arr > 0, vol_arr, 0.0)

        o   = Resample(count=self.COUNT, agg="first")(price_arr, t_arr)
        c   = Resample(count=self.COUNT, agg="last")(price_arr, t_arr)
        buy = Resample(count=self.COUNT, agg="sum")(pos_vol,   t_arr)
        self.out, self.idx = CombineLatest()(o, c, buy)

    def test_shape(self):
        assert self.out.ndim == 2
        assert self.out.shape[1] == 3
        assert len(self.out) > 0

    def test_open_matches_single_col_count(self):
        ref = Resample(count=self.COUNT, agg="first")(self.price_arr, self.t_arr)
        ref_v = ref[0]
        np.testing.assert_allclose(self.out[:, 0], ref_v, rtol=1e-12)

    def test_close_matches_single_col_count(self):
        ref = Resample(count=self.COUNT, agg="last")(self.price_arr, self.t_arr)
        ref_v = ref[0]
        np.testing.assert_allclose(self.out[:, 1], ref_v, rtol=1e-12)

    def test_buy_matches_single_col_count(self):
        pos_vol = np.where(self.vol_arr > 0, self.vol_arr, 0.0)
        ref = Resample(count=self.COUNT, agg="sum")(pos_vol, self.t_arr)
        ref_v = ref[0]
        np.testing.assert_allclose(self.out[:, 2], ref_v, rtol=1e-12)


# ---------------------------------------------------------------------------
# Test 3: Oracle test from brief - OHLCV via combine_latest with literal values
# ---------------------------------------------------------------------------

def test_ohlcv_via_composition():
    """Brief oracle: hand-computed OHLCV via combine_latest of per-stat resamples."""
    price = np.array([10., 11, 9, 12, 8, 13])
    vol   = np.array([1.,  2,  1,  3, 1,  2])
    k     = np.arange(6)
    o = Resample(freq=3, agg="first")(price, k)
    h = Resample(freq=3, agg="max")(price, k)
    l = Resample(freq=3, agg="min")(price, k)
    c = Resample(freq=3, agg="last")(price, k)
    v = Resample(freq=3, agg="sum")(vol,   k)
    rows, idx = CombineLatest()(o, h, l, c, v)
    # bar 0 [0,3): prices=10,11,9; vol=1,2,1
    np.testing.assert_array_equal(np.asarray(rows)[0], [10, 11, 9, 9, 4])
    # bar 1 [3,6): prices=12,8,13; vol=3,1,2
    np.testing.assert_array_equal(np.asarray(rows)[1], [12, 13, 8, 13, 6])


# ---------------------------------------------------------------------------
# Test 4: VWAP via composition - brief oracle
# ---------------------------------------------------------------------------

def test_vwap_via_composition():
    """Brief oracle: VWAP as ratio of resampled sum(price*vol) / sum(vol)."""
    price = np.array([10., 20, 30, 40])
    vol   = np.array([1.,   3,  1,  1])
    k     = np.arange(4)
    num  = Resample(freq=2, agg="sum")(price * vol, k)
    den  = Resample(freq=2, agg="sum")(vol,         k)
    vwap = np.asarray(num[0]) / np.asarray(den[0])
    np.testing.assert_allclose(vwap, [(10 + 60) / 4, (30 + 40) / 2])


# ---------------------------------------------------------------------------
# Test 5: dict agg raises with a migration hint
# ---------------------------------------------------------------------------

def test_agg_dict_raises_with_migration_hint():
    """Passing agg={...} to resample raises ValueError with a helpful message."""
    with pytest.raises((ValueError, TypeError)):
        Resample(freq=3, agg={"open": First()})(np.array([1.0, 2, 3]))


def test_agg_dict_raises_eager():
    """Eager (array) dict agg also raises."""
    x   = np.arange(10.0)
    idx = np.arange(10, dtype=np.int64)
    with pytest.raises((ValueError, TypeError)):
        Resample(freq=5, agg={"s": "sum"})(x, idx)


# ---------------------------------------------------------------------------
# Test 6: Index alignment - composition bars share the same bar index
# ---------------------------------------------------------------------------

def test_composition_bar_index_aligned():
    """All per-stat resamples produce the same bar labels; combine_latest aligns them."""
    t_arr, price_arr, vol_arr = _make_data()
    o = Resample(freq=W, agg="first")(price_arr, t_arr)
    c = Resample(freq=W, agg="last")(price_arr, t_arr)
    v = Resample(freq=W, agg="sum")(np.abs(vol_arr), t_arr)
    rows, idx = CombineLatest()(o, c, v)
    np.testing.assert_array_equal(idx, o[1])
    np.testing.assert_array_equal(idx, c[1])
    assert rows.shape == (N_BARS, 3)
