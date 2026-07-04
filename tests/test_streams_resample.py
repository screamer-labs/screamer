import math
import numpy as np
import pytest

from screamer.streams import resample, resample_iter


def test_resample_by_key_last_left_label():
    # width 10, keys 0..25 -> buckets [0,10),[10,20),[20,30)
    keys = np.array([0, 3, 10, 12, 20], dtype=np.int64)
    vals = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    k, v = resample(keys, vals, width=10, agg="last")
    np.testing.assert_array_equal(k, [0, 10, 20])
    np.testing.assert_array_equal(v, [2.0, 4.0, 5.0])   # last in each bucket; [20,30) trailing


def test_resample_by_key_right_label():
    keys = np.array([0, 3, 10, 12], dtype=np.int64)
    vals = np.array([1.0, 2.0, 3.0, 4.0])
    k, v = resample(keys, vals, width=10, agg="sum", label="right")
    np.testing.assert_array_equal(k, [10, 20])          # right = bucket end
    np.testing.assert_array_equal(v, [3.0, 7.0])


def test_resample_by_key_origin():
    keys = np.array([2, 7, 12], dtype=np.int64)
    vals = np.array([1.0, 2.0, 3.0])
    # origin=2, width=5 -> buckets [2,7),[7,12),[12,17)
    k, v = resample(keys, vals, width=5, origin=2, agg="first")
    np.testing.assert_array_equal(k, [2, 7, 12])
    np.testing.assert_array_equal(v, [1.0, 2.0, 3.0])


def test_resample_aggregations():
    keys = np.array([0, 1, 2, 3], dtype=np.int64)
    vals = np.array([4.0, 2.0, 8.0, 6.0])
    # single bucket width 10
    assert resample(keys, vals, width=10, agg="min")[1].tolist() == [2.0]
    assert resample(keys, vals, width=10, agg="max")[1].tolist() == [8.0]
    assert resample(keys, vals, width=10, agg="sum")[1].tolist() == [20.0]
    assert resample(keys, vals, width=10, agg="count")[1].tolist() == [4.0]
    assert resample(keys, vals, width=10, agg="mean")[1].tolist() == [5.0]
    assert resample(keys, vals, width=10, agg="first")[1].tolist() == [4.0]


def test_resample_ohlc_width4():
    keys = np.array([0, 1, 2, 3], dtype=np.int64)
    vals = np.array([4.0, 2.0, 8.0, 6.0])
    k, v = resample(keys, vals, width=10, agg="ohlc")
    assert v.shape == (1, 4)
    np.testing.assert_array_equal(v[0], [4.0, 8.0, 2.0, 6.0])   # open,high,low,close


def test_resample_nan_ignore():
    keys = np.array([0, 1, 2], dtype=np.int64)
    vals = np.array([np.nan, 4.0, np.nan])
    # bucket has events but one finite value
    assert resample(keys, vals, width=10, agg="mean")[1].tolist() == [4.0]
    assert resample(keys, vals, width=10, agg="count")[1].tolist() == [1.0]


def test_resample_all_nan_bucket_emits_nan():
    keys = np.array([0, 1], dtype=np.int64)
    vals = np.array([np.nan, np.nan])
    k, v = resample(keys, vals, width=10, agg="mean")
    assert len(k) == 1 and math.isnan(v[0])
    # sum of all-nan bucket is 0.0, count is 0
    assert resample(keys, vals, width=10, agg="sum")[1].tolist() == [0.0]
    assert resample(keys, vals, width=10, agg="count")[1].tolist() == [0.0]


def test_resample_sparse_skips_empty_buckets():
    keys = np.array([0, 100], dtype=np.int64)   # width 10: buckets 0 and 10, nothing between
    vals = np.array([1.0, 2.0])
    k, v = resample(keys, vals, width=10, agg="last")
    np.testing.assert_array_equal(k, [0, 100])   # only the two non-empty buckets


def test_resample_by_count():
    keys = np.array([10, 20, 30, 40, 50], dtype=np.int64)
    vals = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    k, v = resample(keys, vals, count=2, agg="sum")
    # buckets [1,2],[3,4],[5] (trailing partial); left label = first key of bucket
    np.testing.assert_array_equal(k, [10, 30, 50])
    np.testing.assert_array_equal(v, [3.0, 7.0, 5.0])


def test_resample_by_count_right_label():
    keys = np.array([10, 20, 30, 40], dtype=np.int64)
    vals = np.array([1.0, 2.0, 3.0, 4.0])
    k, v = resample(keys, vals, count=2, agg="last", label="right")
    np.testing.assert_array_equal(k, [20, 40])   # right = last key of bucket


def test_resample_iter_matches_batch():
    keys = np.array([0, 3, 10, 12, 20], dtype=np.int64)
    vals = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    bk, bv = resample(keys, vals, width=10, agg="mean")
    events = list(zip(keys.tolist(), vals.tolist()))
    got = list(resample_iter(events, width=10, agg="mean"))
    assert [k for k, _ in got] == bk.tolist()
    assert [v for _, v in got] == bv.tolist()


def test_resample_validation_errors():
    keys = np.array([0, 1], dtype=np.int64); vals = np.array([1.0, 2.0])
    with pytest.raises(ValueError, match="exactly one"):
        resample(keys, vals)                      # neither width nor count
    with pytest.raises(ValueError, match="exactly one"):
        resample(keys, vals, width=10, count=2)   # both
    with pytest.raises(ValueError, match="agg"):
        resample(keys, vals, width=10, agg="nope")
    with pytest.raises(ValueError, match="label"):
        resample(keys, vals, width=10, label="middle")
    with pytest.raises(ValueError, match="1-D"):
        resample(keys, np.zeros((2, 2)), width=10)   # wide input


def test_resample_by_key_negative_keys():
    # Floor-division (NOT C-style truncation) for negative keys. width=10, origin=0:
    # floor(-15/10)=-2 -> label -20; floor(-5/10)=-1 -> label -10; floor(5/10)=0 -> 0.
    # C truncation (int(-15/10)=-1) would merge -15 and -5 into the wrong bucket.
    # Pins the exact parity the C++ engine's floordiv must match.
    keys = np.array([-15, -5, 5], dtype=np.int64)
    vals = np.array([1.0, 2.0, 3.0])
    k, v = resample(keys, vals, width=10, agg="last")
    np.testing.assert_array_equal(k, [-20, -10, 0])
    np.testing.assert_array_equal(v, [1.0, 2.0, 3.0])


def test_resample_by_count_iter_and_nan():
    # by-count iter matches batch, and NaN is ignored inside a count bucket
    keys = np.array([10, 20, 30, 40], dtype=np.int64)
    vals = np.array([np.nan, 2.0, np.nan, 4.0])
    bk, bv = resample(keys, vals, count=2, agg="mean")
    np.testing.assert_array_equal(bk, [10, 30])
    np.testing.assert_array_equal(bv, [2.0, 4.0])
    events = list(zip(keys.tolist(), vals.tolist()))
    got = list(resample_iter(events, count=2, agg="mean"))
    assert [k for k, _ in got] == bk.tolist()
    assert [v for _, v in got] == bv.tolist()


def test_resample_empty_input():
    k, v = resample(np.array([], dtype=np.int64), np.array([]), width=10, agg="sum")
    assert len(k) == 0 and len(v) == 0


def test_resample_nonpositive_width_count_rejected():
    keys = np.array([0, 1], dtype=np.int64); vals = np.array([1.0, 2.0])
    # width=0 would otherwise reach the engine floordiv(_, 0) -> SIGFPE crash
    with pytest.raises(ValueError, match="width must be positive"):
        resample(keys, vals, width=0)
    with pytest.raises(ValueError, match="width must be positive"):
        resample(keys, vals, width=-5)
    with pytest.raises(ValueError, match="count must be"):
        resample(keys, vals, count=0)
