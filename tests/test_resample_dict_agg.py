"""Tests for resample(agg={name: str|functor, ...}) - Task 6 dict-agg branch."""
import numpy as np
import pytest
from screamer import ExpandingSkew
from screamer.streams import resample


def test_dict_agg_labelled_columns():
    """Dict agg returns a Stream with columns = tuple(dict.keys())."""
    x = np.arange(20.0)
    idx = np.arange(20, dtype=np.int64)
    bars = resample(x, idx, every=5, agg={"total": "sum", "skew": ExpandingSkew()})
    assert tuple(bars.columns) == ("total", "skew")
    total, _ = resample(x, idx, every=5, agg="sum")
    np.testing.assert_allclose(
        bars["total"],
        total.values if hasattr(total, "values") else total,
    )


def test_dict_agg_column_order_matches_insertion_order():
    """Column order follows dict insertion order."""
    x = np.arange(10.0)
    idx = np.arange(10, dtype=np.int64)
    bars = resample(x, idx, every=5, agg={"z": "last", "a": "first", "m": "mean"})
    assert tuple(bars.columns) == ("z", "a", "m")


def test_dict_agg_matches_single_agg_values():
    """Each column in a dict agg matches its single-agg equivalent."""
    x = np.arange(20.0)
    idx = np.arange(20, dtype=np.int64)
    bars = resample(x, idx, every=5, agg={"s": "sum", "mn": "min", "mx": "max"})

    for name, agg_str in [("s", "sum"), ("mn", "min"), ("mx", "max")]:
        ref_vals, _ = resample(x, idx, every=5, agg=agg_str)
        np.testing.assert_allclose(bars[name], ref_vals)


def test_dict_agg_bar_index_matches_single_agg():
    """Dict agg produces the same bar labels as the equivalent single-agg."""
    x = np.arange(20.0)
    idx = np.arange(20, dtype=np.int64)
    bars = resample(x, idx, every=5, agg={"s": "sum", "c": "count"})
    ref, ref_idx = resample(x, idx, every=5, agg="sum")
    np.testing.assert_array_equal(bars.index, ref_idx)


def test_dict_agg_rejects_ohlc_sub_agg():
    """Using 'ohlc' inside a dict agg raises ValueError (multi-column v1 restriction)."""
    x = np.arange(10.0)
    idx = np.arange(10, dtype=np.int64)
    with pytest.raises(ValueError, match="ohlc"):
        resample(x, idx, every=5, agg={"o": "ohlc"})


def test_dict_agg_rejects_unknown_string_sub_agg():
    """Unknown string sub-agg raises ValueError."""
    x = np.arange(10.0)
    idx = np.arange(10, dtype=np.int64)
    with pytest.raises(ValueError):
        resample(x, idx, every=5, agg={"x": "bad_agg"})


def test_dict_agg_rejects_empty_dict():
    """Empty dict raises ValueError."""
    x = np.arange(10.0)
    idx = np.arange(10, dtype=np.int64)
    with pytest.raises(ValueError):
        resample(x, idx, every=5, agg={})


def test_dict_agg_single_entry():
    """Single-entry dict returns 2-D Stream with one column."""
    x = np.arange(10.0)
    idx = np.arange(10, dtype=np.int64)
    bars = resample(x, idx, every=5, agg={"total": "sum"})
    assert tuple(bars.columns) == ("total",)
    assert bars.values.ndim == 2
    assert bars.values.shape[1] == 1


def test_dict_agg_count_bucketing():
    """Dict agg works with count= bucketing (not just every=)."""
    x = np.arange(12.0)
    idx = np.arange(12, dtype=np.int64)
    bars = resample(x, idx, count=4, agg={"s": "sum", "f": "first"})
    assert tuple(bars.columns) == ("s", "f")
    ref_s, _ = resample(x, idx, count=4, agg="sum")
    ref_f, _ = resample(x, idx, count=4, agg="first")
    np.testing.assert_allclose(bars["s"], ref_s)
    np.testing.assert_allclose(bars["f"], ref_f)
