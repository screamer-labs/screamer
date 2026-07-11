import math
import numpy as np
import pytest

from screamer import ExpandingSum
from screamer.streams import resample, Stream
from screamer.dag import is_node


# ---------------------------------------------------------------------------
# Basic by-key (every=) tests
# ---------------------------------------------------------------------------

def test_resample_by_key_last_left_label():
    # every=10, keys 0..20 -> buckets [0,10),[10,20),[20,30)
    keys = np.array([0, 3, 10, 12, 20], dtype=np.int64)
    vals = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    v, k = resample(vals, keys, every=10, agg="last")
    np.testing.assert_array_equal(v, [2.0, 4.0, 5.0])   # last in each bucket
    np.testing.assert_array_equal(k, [0, 10, 20])        # bar labels (left)


def test_resample_by_key_right_label():
    keys = np.array([0, 3, 10, 12], dtype=np.int64)
    vals = np.array([1.0, 2.0, 3.0, 4.0])
    v, k = resample(vals, keys, every=10, agg="sum", label="right")
    np.testing.assert_array_equal(k, [10, 20])           # right = bucket end
    np.testing.assert_array_equal(v, [3.0, 7.0])


def test_resample_by_key_origin():
    keys = np.array([2, 7, 12], dtype=np.int64)
    vals = np.array([1.0, 2.0, 3.0])
    # origin=2, every=5 -> buckets [2,7),[7,12),[12,17)
    v, k = resample(vals, keys, every=5, origin=2, agg="first")
    np.testing.assert_array_equal(k, [2, 7, 12])
    np.testing.assert_array_equal(v, [1.0, 2.0, 3.0])


def test_resample_aggregations():
    keys = np.array([0, 1, 2, 3], dtype=np.int64)
    vals = np.array([4.0, 2.0, 8.0, 6.0])
    # single bucket every=10
    assert resample(vals, keys, every=10, agg="min")[0].tolist() == [2.0]
    assert resample(vals, keys, every=10, agg="max")[0].tolist() == [8.0]
    assert resample(vals, keys, every=10, agg="sum")[0].tolist() == [20.0]
    assert resample(vals, keys, every=10, agg="count")[0].tolist() == [4.0]
    assert resample(vals, keys, every=10, agg="mean")[0].tolist() == [5.0]
    assert resample(vals, keys, every=10, agg="first")[0].tolist() == [4.0]


def test_resample_ohlc_4col():
    keys = np.array([0, 1, 2, 3], dtype=np.int64)
    vals = np.array([4.0, 2.0, 8.0, 6.0])
    v, k = resample(vals, keys, every=10, agg="ohlc")
    assert v.shape == (1, 4)
    np.testing.assert_array_equal(v[0], [4.0, 8.0, 2.0, 6.0])   # open,high,low,close


def test_resample_nan_ignore():
    keys = np.array([0, 1, 2], dtype=np.int64)
    vals = np.array([np.nan, 4.0, np.nan])
    # bucket has events but one finite value
    assert resample(vals, keys, every=10, agg="mean")[0].tolist() == [4.0]
    assert resample(vals, keys, every=10, agg="count")[0].tolist() == [1.0]


def test_resample_all_nan_bucket_emits_nan():
    keys = np.array([0, 1], dtype=np.int64)
    vals = np.array([np.nan, np.nan])
    v, k = resample(vals, keys, every=10, agg="mean")
    assert len(v) == 1 and math.isnan(v[0])
    # sum of all-nan bucket is 0.0, count is 0
    assert resample(vals, keys, every=10, agg="sum")[0].tolist() == [0.0]
    assert resample(vals, keys, every=10, agg="count")[0].tolist() == [0.0]


def test_resample_sparse_skips_empty_buckets():
    keys = np.array([0, 100], dtype=np.int64)   # every=10: only two non-empty buckets
    vals = np.array([1.0, 2.0])
    v, k = resample(vals, keys, every=10, agg="last")
    np.testing.assert_array_equal(k, [0, 100])   # only the two non-empty buckets
    np.testing.assert_array_equal(v, [1.0, 2.0])


# ---------------------------------------------------------------------------
# By-count tests
# ---------------------------------------------------------------------------

def test_resample_by_count():
    keys = np.array([10, 20, 30, 40, 50], dtype=np.int64)
    vals = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    v, k = resample(vals, keys, count=2, agg="sum")
    # buckets [1,2],[3,4],[5] (trailing partial); left label = first key of bucket
    np.testing.assert_array_equal(k, [10, 30, 50])
    np.testing.assert_array_equal(v, [3.0, 7.0, 5.0])


def test_resample_by_count_right_label():
    keys = np.array([10, 20, 30, 40], dtype=np.int64)
    vals = np.array([1.0, 2.0, 3.0, 4.0])
    v, k = resample(vals, keys, count=2, agg="last", label="right")
    np.testing.assert_array_equal(k, [20, 40])   # right = last key of bucket


# ---------------------------------------------------------------------------
# Lazy (iterator) tests - resample dispatches on input type
# ---------------------------------------------------------------------------

def test_resample_lazy_equals_batch_every():
    vals = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
    idx = np.array([0, 1, 2, 10, 11, 20, 21])
    batch = resample(vals, idx, every=10, agg="mean")   # a Stream, unpackable
    bv, bk = batch.values, batch.index
    gen = ((float(v), int(k)) for v, k in zip(vals, idx))
    out = resample(gen, every=10, agg="mean")
    assert hasattr(out, "__next__") and not isinstance(out, tuple)
    rows = list(out)                                     # [(bar_value, bar_label), ...]
    np.testing.assert_allclose([r[0] for r in rows], np.asarray(bv), equal_nan=True)
    np.testing.assert_array_equal([r[1] for r in rows], np.asarray(bk))


def test_resample_lazy_equals_batch_count_and_nan():
    vals = np.array([1.0, np.nan, 3.0, 4.0, 5.0, 6.0])
    idx = np.array([0, 1, 2, 3, 4, 5])
    batch = resample(vals, idx, count=2, agg="mean")
    bv, bk = batch.values, batch.index
    gen = ((float(v), int(k)) for v, k in zip(vals, idx))
    rows = list(resample(gen, count=2, agg="mean"))
    np.testing.assert_allclose([r[0] for r in rows], np.asarray(bv), equal_nan=True)
    np.testing.assert_array_equal([r[1] for r in rows], np.asarray(bk))


def test_resample_lazy_functor_agg_equals_batch():
    vals = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    idx = np.array([0, 1, 2, 3, 4])
    batch = resample(vals, idx, count=2, agg=ExpandingSum())
    bv, bk = batch.values, batch.index
    gen = ((float(v), int(k)) for v, k in zip(vals, idx))
    rows = list(resample(gen, count=2, agg=ExpandingSum()))
    np.testing.assert_allclose([r[0] for r in rows], np.asarray(bv), equal_nan=True)
    np.testing.assert_array_equal([r[1] for r in rows], np.asarray(bk))


def test_resample_lazy_rejects_multicolumn_aggs():
    from screamer import ExpandingSum
    # The reject fires eagerly at the resample(...) call (before any iteration),
    # so assert on the call itself, not on list(...) of a would-be iterator.
    for agg in ("ohlcv", "ohlcv2", {"x": ExpandingSum()}):
        with pytest.raises(ValueError):
            resample((e for e in ((1.0, 0), (2.0, 1))), count=2, agg=agg)


def test_resample_lazy_is_lazy():
    """The lazy path must pull events on demand, not eagerly on construction."""
    pulled = []

    def spy():
        for i, v in enumerate([1.0, 2.0, 3.0, 4.0]):
            pulled.append(v)
            yield (v, i)

    it = resample(spy(), count=2, agg="last")
    assert pulled == []            # nothing consumed before first next()
    first = next(it)
    assert pulled == [1.0, 2.0]    # exactly the first bucket's events consumed
    assert first == (2.0, 0)       # count=2, last, label="left" -> value 2.0 at index 0


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------

def test_resample_validation_errors():
    vals = np.array([1.0, 2.0])
    keys = np.array([0, 1], dtype=np.int64)
    with pytest.raises(ValueError, match="exactly one"):
        resample(vals)                              # none of freq/every/count
    with pytest.raises(ValueError, match="exactly one"):
        resample(vals, keys, every=10, count=2)    # both
    with pytest.raises(ValueError, match="agg"):
        resample(vals, keys, every=10, agg="nope")
    with pytest.raises(ValueError, match="label"):
        resample(vals, keys, every=10, label="middle")
    with pytest.raises(ValueError, match="1-D"):
        resample(np.zeros((2, 2)), every=10)        # wide input


def test_resample_nonpositive_every_count_rejected():
    vals = np.array([1.0, 2.0])
    # every=0 would otherwise reach the engine floordiv(_, 0) -> SIGFPE crash
    with pytest.raises(ValueError, match="every must be positive"):
        resample(vals, every=0)
    with pytest.raises(ValueError, match="every must be positive"):
        resample(vals, every=-5)
    with pytest.raises(ValueError, match="count must be"):
        resample(vals, count=0)


# ---------------------------------------------------------------------------
# Negative index (floor-division pinning)
# ---------------------------------------------------------------------------

def test_resample_by_key_negative_keys():
    # Floor-division (NOT C-style truncation) for negative keys. every=10, origin=0:
    # floor(-15/10)=-2 -> label -20; floor(-5/10)=-1 -> label -10; floor(5/10)=0 -> 0.
    # C truncation (int(-15/10)=-1) would merge -15 and -5 into the wrong bucket.
    # Pins the exact parity the C++ engine's floordiv must match.
    keys = np.array([-15, -5, 5], dtype=np.int64)
    vals = np.array([1.0, 2.0, 3.0])
    v, k = resample(vals, keys, every=10, agg="last")
    np.testing.assert_array_equal(v, [1.0, 2.0, 3.0])
    np.testing.assert_array_equal(k, [-20, -10, 0])


# ---------------------------------------------------------------------------
# Empty input
# ---------------------------------------------------------------------------

def test_resample_empty_input():
    v, k = resample(np.array([]), every=10, agg="sum")
    assert len(v) == 0 and len(k) == 0


# ---------------------------------------------------------------------------
# Polymorphic regime: raw / Stream / Node + positional
# ---------------------------------------------------------------------------

def test_resample_raw_returns_stream_with_real_labels():
    """Raw input now returns a Stream; index holds bar labels (never None).

    Stream is unpackable as (values, index) for backward-compatible unpacking,
    so ``out_v, out_k = result`` still works.
    """
    vals = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    keys = np.array([0, 3, 10, 12, 20], dtype=np.int64)
    result = resample(vals, keys, every=10, agg="last")
    assert isinstance(result, Stream)
    out_v, out_k = result            # Stream unpacks as (values, index)
    assert out_k is not None         # bar labels are always a real array
    np.testing.assert_array_equal(out_v, [2.0, 4.0, 5.0])
    np.testing.assert_array_equal(out_k, [0, 10, 20])


def test_resample_stream_input_returns_stream():
    """Stream input returns a Stream; index holds the bar labels."""
    vals = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    keys = np.array([0, 3, 10, 12, 20], dtype=np.int64)
    s = Stream(vals, keys)
    result = resample(s, every=10, agg="last")
    assert isinstance(result, Stream)
    np.testing.assert_array_equal(result.values, [2.0, 4.0, 5.0])
    np.testing.assert_array_equal(result.index, [0, 10, 20])


def test_resample_node_input_returns_node():
    """Node input returns a Node (graph-mode)."""
    from screamer import Input
    x = Input("x")
    node = resample(x, every=10, agg="last")
    assert is_node(node)


def test_resample_positional_input_uses_row_positions():
    """No index (positional) input: resample by row number; labels are real (not None)."""
    vals = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    # every=2 on row positions 0,1,2,3,4 -> buckets [0,2),[2,4),[4,6)
    # left labels: 0, 2, 4; last of each bucket: 2.0, 4.0, 5.0
    v, k = resample(vals, every=2, agg="last")
    assert k is not None           # real labels, never None
    np.testing.assert_array_equal(k, [0, 2, 4])
    np.testing.assert_array_equal(v, [2.0, 4.0, 5.0])


def test_resample_raw_stream_node_mirror():
    """Raw, Stream, and Node inputs all produce the same output values and labels."""
    from screamer import Input, Dag
    vals = np.array([1.0, 2.0, 3.0, 4.0])
    keys = np.array([0, 5, 10, 15], dtype=np.int64)

    # Raw
    rv, rk = resample(vals, keys, every=10, agg="sum")

    # Stream
    s = Stream(vals, keys)
    stream_out = resample(s, every=10, agg="sum")

    # Node (via Dag)
    x = Input("x")
    dag = Dag(inputs=[x], outputs=[resample(x, every=10, agg="sum")])
    dag_v, dag_k = dag((vals, keys))        # (values, index) feed; values-first result

    np.testing.assert_array_equal(rv, stream_out.values)
    np.testing.assert_array_equal(rk, stream_out.index)
    np.testing.assert_array_equal(rv, dag_v.reshape(-1))
    np.testing.assert_array_equal(rk, dag_k)


def test_resample_fractional_index_raises():
    """resample rides the int64-indexed engine; a fractional index must raise
    (both batch and lazy) instead of being silently floored."""
    import pytest
    vals, idx = np.array([1.0, 2.0, 3.0, 4.0]), np.array([0.0, 1.5, 2.0, 3.0])
    with pytest.raises(TypeError):
        resample(vals, idx, every=2, agg="mean")                          # batch
    with pytest.raises(TypeError):
        list(resample(((float(v), 0.5 * v) for v in range(4)), count=2, agg="mean"))  # lazy


# ---------------------------------------------------------------------------
# freq= parameter (Task 1)
# ---------------------------------------------------------------------------

def test_freq_no_index_equals_count():
    v = np.array([1.0, 2, 3, 4, 5, 6])
    old = resample(v, count=2, agg="mean")
    new = resample(v, freq=2, agg="mean")
    np.testing.assert_array_equal(np.asarray(new.values), np.asarray(old.values))
    np.testing.assert_array_equal(np.asarray(new.index), np.asarray(old.index))


def test_freq_integer_index_equals_every():
    v = np.array([1.0, 2, 3, 4, 5])
    k = np.array([0, 1, 2, 10, 11])
    old = resample(v, k, every=10, agg="sum")
    new = resample(v, k, freq=10, agg="sum")
    np.testing.assert_array_equal(np.asarray(new.values), np.asarray(old.values))
    np.testing.assert_array_equal(np.asarray(new.index), np.asarray(old.index))


def test_freq_stream_uses_its_own_index_not_kwarg():
    # A Stream carries its own index, so freq reads span-vs-count from s.index,
    # not the index= kwarg. Stream(v, int index) + freq -> span (== every).
    v = np.array([1.0, 2, 3, 4, 5])
    k = np.array([0, 1, 2, 10, 11])
    old = resample(v, k, every=10, agg="sum")
    new = resample(Stream(v, k), freq=10, agg="sum")
    np.testing.assert_array_equal(np.asarray(new.values), np.asarray(old.values))
    np.testing.assert_array_equal(np.asarray(new.index), np.asarray(old.index))
    # a Stream with no index -> count mode
    cnt = resample(Stream(v), freq=2, agg="sum")
    ref = resample(v, count=2, agg="sum")
    np.testing.assert_array_equal(np.asarray(cnt.values), np.asarray(ref.values))


def test_freq_rejects_nonpositive_and_missing():
    with pytest.raises((TypeError, ValueError)):
        resample(np.array([1.0, 2, 3]))                 # freq is required
    with pytest.raises(ValueError):
        resample(np.array([1.0, 2, 3]), freq=0)         # must be positive


# ---------------------------------------------------------------------------
# datetime64 index + offset string / timedelta freq (Task 2)
# ---------------------------------------------------------------------------

def test_freq_datetime_offset_and_timedelta_agree():
    t = np.array(["2020-01-01T00:00:00", "2020-01-01T00:00:30",
                  "2020-01-01T00:01:10"], dtype="datetime64[s]")
    v = np.array([1.0, 2.0, 3.0])
    a = resample(v, t, freq="1min", agg="sum")
    b = resample(v, t, freq=np.timedelta64(60, "s"), agg="sum")
    np.testing.assert_array_equal(np.asarray(a.values), np.asarray(b.values))


def test_freq_timedelta_on_integer_index_raises():
    with pytest.raises((TypeError, ValueError)):
        resample(np.array([1.0, 2, 3]), np.array([0, 1, 2]), freq="1min", agg="sum")


def test_freq_datetime_offset_string_minute_bars():
    # Two ticks in [00:00, 00:01), one tick in [00:01, 00:02)
    t = np.array(["2020-01-01T00:00:00", "2020-01-01T00:00:30",
                  "2020-01-01T00:01:10"], dtype="datetime64[s]")
    v = np.array([1.0, 2.0, 3.0])
    result = resample(v, t, freq="1min", agg="sum")
    np.testing.assert_array_equal(np.asarray(result.values), [3.0, 3.0])


def test_freq_datetime_timedelta_object():
    import datetime
    t = np.array(["2020-01-01T00:00:00", "2020-01-01T00:00:30",
                  "2020-01-01T00:01:10"], dtype="datetime64[s]")
    v = np.array([1.0, 2.0, 3.0])
    result = resample(v, t, freq=datetime.timedelta(minutes=1), agg="sum")
    np.testing.assert_array_equal(np.asarray(result.values), [3.0, 3.0])


def test_freq_datetime_offset_units():
    # "T" is alias for "min"
    t = np.array(["2020-01-01T00:00:00", "2020-01-01T00:00:30",
                  "2020-01-01T00:01:10"], dtype="datetime64[s]")
    v = np.array([1.0, 2.0, 3.0])
    a = resample(v, t, freq="T", agg="sum")
    b = resample(v, t, freq="1min", agg="sum")
    np.testing.assert_array_equal(np.asarray(a.values), np.asarray(b.values))


def test_freq_datetime_nanosecond_resolution():
    # index in ns; "1min" -> span of 60_000_000_000 ns
    t = np.array(["2020-01-01T00:00:00", "2020-01-01T00:00:30",
                  "2020-01-01T00:01:10"], dtype="datetime64[ns]")
    v = np.array([1.0, 2.0, 3.0])
    result = resample(v, t, freq="1min", agg="sum")
    np.testing.assert_array_equal(np.asarray(result.values), [3.0, 3.0])


def test_freq_int_on_datetime_index_raises():
    t = np.array(["2020-01-01T00:00:00", "2020-01-01T00:00:30"], dtype="datetime64[s]")
    with pytest.raises((TypeError, ValueError)):
        resample(np.array([1.0, 2.0]), t, freq=60, agg="sum")


def test_freq_datetime_unsupported_offset_raises():
    t = np.array(["2020-01-01", "2020-02-01"], dtype="datetime64[D]")
    with pytest.raises((NotImplementedError, ValueError)):
        resample(np.array([1.0, 2.0]), t, freq="1M", agg="sum")


def test_freq_datetime_offset_not_exact_multiple_raises():
    # A freq that is not a whole multiple of the index resolution must raise, not
    # silently truncate: 1500ms into a datetime64[s] index is 1.5s (non-integer).
    t_s = np.array(["2020-01-01T00:00:00"], dtype="datetime64[s]")
    with pytest.raises(ValueError):
        resample(np.array([1.0]), t_s,
                 freq=np.timedelta64(1500, "ms"), agg="sum")


def test_freq_timedelta64_on_integer_index_raises_clearly():
    # np.timedelta64 subclasses np.integer, so guard against it being treated as a
    # numeric span on an integer index; must raise a clear TypeError.
    with pytest.raises(TypeError):
        resample(np.array([1.0, 2, 3]), np.array([0, 1, 2]),
                 freq=np.timedelta64(1, "m"), agg="sum")
