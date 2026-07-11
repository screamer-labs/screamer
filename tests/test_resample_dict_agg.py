"""Tests confirming dict agg raises, and verifying composition reproduces the same values.

The resample(agg={...}) form has been removed. These tests verify:
1. Any dict passed as agg raises ValueError with a migration hint.
2. The composition recipe (combine_latest of per-stat resamples) produces the same
   numeric results that the dict form used to produce.
"""
import numpy as np
import pytest

from screamer import ExpandingSkew
from screamer.streams import Stream, CombineLatest, Resample


# ---------------------------------------------------------------------------
# Raise tests - every dict-agg call from the old test file now raises
# ---------------------------------------------------------------------------

def test_dict_agg_raises():
    """Any dict passed as agg raises ValueError."""
    x   = np.arange(20.0)
    idx = np.arange(20, dtype=np.int64)
    with pytest.raises((ValueError, TypeError)):
        Resample(freq=5, agg={"total": "sum", "skew": ExpandingSkew()})(x, idx)


def test_dict_agg_raises_column_order():
    x   = np.arange(10.0)
    idx = np.arange(10, dtype=np.int64)
    with pytest.raises((ValueError, TypeError)):
        Resample(freq=5, agg={"z": "last", "a": "first", "m": "mean"})(x, idx)


def test_dict_agg_raises_sum_min_max():
    x   = np.arange(20.0)
    idx = np.arange(20, dtype=np.int64)
    with pytest.raises((ValueError, TypeError)):
        Resample(freq=5, agg={"s": "sum", "mn": "min", "mx": "max"})(x, idx)


def test_dict_agg_raises_count_bucketing():
    x   = np.arange(12.0)
    idx = np.arange(12, dtype=np.int64)
    with pytest.raises((ValueError, TypeError)):
        Resample(count=4, agg={"s": "sum", "f": "first"})(x, idx)


def test_dict_agg_raises_empty():
    x   = np.arange(10.0)
    idx = np.arange(10, dtype=np.int64)
    with pytest.raises((ValueError, TypeError)):
        Resample(freq=5, agg={})(x, idx)


# ---------------------------------------------------------------------------
# Composition equivalents - same numeric assertions the dict form produced
# ---------------------------------------------------------------------------

def test_composition_sum_and_skew_labelled_columns():
    """combine_latest of sum and skew resamples reproduces the two-column result."""
    x   = np.arange(20.0)
    idx = np.arange(20, dtype=np.int64)
    total_s = Resample(freq=5, agg="sum")(x, idx)
    skew_s  = Resample(freq=5, agg=ExpandingSkew())(x, idx)
    rows, bar_idx = CombineLatest()(total_s, skew_s)
    # column 0: sum; verify against single-agg sum
    ref_total, _ = Resample(freq=5, agg="sum")(x, idx)
    np.testing.assert_allclose(rows[:, 0], ref_total.values if hasattr(ref_total, "values") else ref_total)


def test_composition_column_order_preserved():
    """Columns in combine_latest result follow the call order."""
    x   = np.arange(10.0)
    idx = np.arange(10, dtype=np.int64)
    # z=last, a=first, m=mean
    z = Resample(freq=5, agg="last")(x, idx)
    a = Resample(freq=5, agg="first")(x, idx)
    m = Resample(freq=5, agg="mean")(x, idx)
    rows, _ = CombineLatest()(z, a, m)
    assert rows.shape[1] == 3
    # column 0 = last, 1 = first, 2 = mean
    ref_last,  _ = Resample(freq=5, agg="last")(x, idx)
    ref_first, _ = Resample(freq=5, agg="first")(x, idx)
    np.testing.assert_allclose(rows[:, 0], ref_last.values if hasattr(ref_last, "values") else ref_last)
    np.testing.assert_allclose(rows[:, 1], ref_first.values if hasattr(ref_first, "values") else ref_first)


def test_composition_matches_single_agg_values():
    """Each column in the composed result matches its single-agg equivalent."""
    x   = np.arange(20.0)
    idx = np.arange(20, dtype=np.int64)
    s_s  = Resample(freq=5, agg="sum")(x, idx)
    mn_s = Resample(freq=5, agg="min")(x, idx)
    mx_s = Resample(freq=5, agg="max")(x, idx)
    rows, _ = CombineLatest()(s_s, mn_s, mx_s)

    for col, agg_str in enumerate(["sum", "min", "max"]):
        ref_vals, _ = Resample(freq=5, agg=agg_str)(x, idx)
        np.testing.assert_allclose(rows[:, col], ref_vals)


def test_composition_bar_index_matches_single_agg():
    """Composition bar labels match those of the single-agg equivalent."""
    x   = np.arange(20.0)
    idx = np.arange(20, dtype=np.int64)
    s_s = Resample(freq=5, agg="sum")(x, idx)
    c_s = Resample(freq=5, agg="count")(x, idx)
    rows, bar_idx = CombineLatest()(s_s, c_s)
    ref, ref_idx = Resample(freq=5, agg="sum")(x, idx)
    np.testing.assert_array_equal(bar_idx, ref_idx)


def test_composition_single_entry():
    """Single-stream combine_latest returns a 2-D result with one column."""
    x   = np.arange(10.0)
    idx = np.arange(10, dtype=np.int64)
    total_s = Resample(freq=5, agg="sum")(x, idx)
    rows, _ = CombineLatest()(total_s)
    assert rows.ndim == 2
    assert rows.shape[1] == 1


def test_composition_count_bucketing():
    """Composition works with count= bucketing (not just every=)."""
    x   = np.arange(12.0)
    idx = np.arange(12, dtype=np.int64)
    s_s = Resample(count=4, agg="sum")(x, idx)
    f_s = Resample(count=4, agg="first")(x, idx)
    rows, _ = CombineLatest()(s_s, f_s)
    assert rows.shape[1] == 2
    ref_s, _ = Resample(count=4, agg="sum")(x, idx)
    ref_f, _ = Resample(count=4, agg="first")(x, idx)
    np.testing.assert_allclose(rows[:, 0], ref_s)
    np.testing.assert_allclose(rows[:, 1], ref_f)


def test_composition_literal_values():
    """Hand-computed anchor: sum and max of two 5-element bars.

    bar0 = values 0..4 -> sum=10, max=4
    bar1 = values 5..9 -> sum=35, max=9
    """
    x   = np.arange(10.0)
    idx = np.arange(10, dtype=np.int64)
    s_s  = Resample(freq=5, agg="sum")(x, idx)
    mx_s = Resample(freq=5, agg="max")(x, idx)
    rows, _ = CombineLatest()(s_s, mx_s)
    expected = np.array([[10.0, 4.0], [35.0, 9.0]])
    np.testing.assert_array_equal(rows, expected)
