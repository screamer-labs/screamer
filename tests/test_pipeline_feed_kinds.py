"""Pipeline index and value kinds (Task 18b: F1, F2, G12, G13, G15).

A datetime64 / timedelta64 index is read as int64 ticks of its own unit in every
entry point (batch, iterator feeds, live push, advance); NaT raises TypeError. A
None value is NaN and an empty event raises ValueError in iterator feeds, as in a
live push. A string that does not read as a number is a TypeError.
"""
import numpy as np
import pandas as pd
import pytest

import screamer as s
from screamer import Input, Merge, Pipeline
from screamer.streams import combine_latest, dropna, merge, resample


def _pipe():
    a = Input("a")
    return Pipeline([a], [s.RollingMean(2)(a)])


def _three_ways(p, vals, idx):
    b = p((vals, np.asarray(idx)))
    lz = list(p(zip(vals, idx)))
    live = p.live()
    for v, k in zip(vals, idx):
        live.push("a", k, v)
    lv = live.result()
    return b, lz, lv


@pytest.mark.parametrize("idx", [
    pd.date_range("2020-01-01", periods=6, freq="s").as_unit("ns"),
    pd.date_range("2020-01-01", periods=6, freq="s"),
    np.arange(6).astype("datetime64[s]"),
    np.array([1, 2, 4, 8, 16, 32], dtype="timedelta64[ms]"),
    pd.timedelta_range(0, periods=6, freq="s"),
], ids=["datetime64ns", "pandas-default", "datetime64s", "timedelta64ms", "pd-timedelta"])
def test_time_index_batch_lazy_live_agree(idx):
    p = _pipe()
    vals = np.array([1.0, 2.0, 4.0, 8.0, 16.0, 32.0])
    (bv, bi), lz, (lv, li) = _three_ways(p, vals, idx)
    ticks = np.asarray(idx).view(np.int64)
    np.testing.assert_array_equal(bi, ticks)
    np.testing.assert_array_equal([e[1] for e in lz], ticks)
    np.testing.assert_array_equal(li, ticks)
    np.testing.assert_array_equal([e[0] for e in lz], bv)
    np.testing.assert_array_equal(lv, bv)
    # Iterating the NumPy array (np.datetime64 scalars) gives the same.
    lz2 = list(p(zip(vals, np.asarray(idx))))
    np.testing.assert_array_equal([e[1] for e in lz2], ticks)


def test_time_index_in_merge_and_advance():
    out = list(Merge()(iter([(1.0, np.datetime64(1, "ns"))]),
                       iter([(2.0, np.datetime64(2, "ns"))])))
    assert [e[1] for e in out] == [1, 2]
    p = _pipe()
    p.live().advance(np.datetime64(5, "ns")).advance(np.datetime64(7, "ns"))
    p.live().advance(np.timedelta64(7, "ms"))


def test_timedelta64_is_ticks_not_integer_branch():
    p = _pipe()
    assert list(p(iter([(1.0, np.timedelta64(3, "ms"))]))) == [(pytest.approx(np.nan, nan_ok=True), 3)]


@pytest.mark.parametrize("nat", [np.datetime64("NaT", "ns"), np.timedelta64("NaT", "ns"), pd.NaT])
def test_nat_index_raises(nat):
    p = _pipe()
    with pytest.raises(TypeError, match="got NaT"):
        list(p(iter([(1.0, nat)])))
    with pytest.raises(TypeError, match="Pipeline: the index of event 0 of input 0: .*NaT"):
        list(p(iter([(1.0, nat)])))
    with pytest.raises(TypeError, match="got NaT"):
        p.live().push("a", nat, 1.0)
    with pytest.raises(TypeError, match="got NaT"):
        p.live().advance(nat)


def test_nat_in_batch_index_raises():
    p = _pipe()
    with pytest.raises(TypeError, match="got NaT"):
        p((np.array([1.0, 2.0]), np.array(["2020-01-01", "NaT"], dtype="M8[ns]")))
    with pytest.raises(TypeError, match="got NaT"):
        p((np.array([1.0, 2.0]), np.array([1, "NaT"], dtype="m8[s]")))


def test_mixed_time_units_in_one_feed_raise():
    p = _pipe()
    with pytest.raises(TypeError, match=r"index is a datetime64\[ns\], but earlier indexes were a datetime64\[s\]"):
        list(p(iter([(1.0, np.datetime64(1, "s")), (2.0, np.datetime64(5, "ns"))])))
    with pytest.raises(TypeError, match="one kind and unit"):
        p.live().push("a", np.datetime64(1, "s"), 1.0).push("a", np.datetime64(5, "ns"), 1.0)
    with pytest.raises(TypeError, match="one kind and unit"):
        list(p(iter([(1.0, np.datetime64(1, "s")), (2.0, np.timedelta64(5, "s"))])))
    with pytest.raises(TypeError, match="one kind and unit"):
        p((np.array([1.0, 2.0]),
           np.array([np.datetime64(1, "s"), np.datetime64(5, "ns")], dtype=object)))
    # A NumPy array has one unit: NumPy converts a list of mixed units.
    _, idx = p((np.array([1.0, 2.0]), [np.datetime64(1, "s"), np.datetime64(5, "ns")]))
    np.testing.assert_array_equal(idx, [10**9, 5])


def _pipe2():
    a, b = Input("a"), Input("b")
    return Pipeline([a, b], [s.Add()(s.CombineLatest()(a, b))])


def test_index_kind_is_one_per_call_or_session():
    p = _pipe2()
    t_s = np.arange(3).astype("datetime64[s]")
    t_ns = np.arange(3).astype("datetime64[ns]")
    v = np.arange(3.0)
    # Batch: across inputs.
    with pytest.raises(TypeError, match="one kind and unit"):
        p((v, t_s), (v, t_ns))
    with pytest.raises(TypeError, match="index is a number, but earlier indexes were a datetime64"):
        p((v, t_s), (v, np.arange(3)))
    with pytest.raises(TypeError, match="one kind and unit"):
        p((v, t_s), v)                       # a bare feed has a positional index
    p((v, t_s), (v, t_s))                    # one unit: fine
    # Iterator feeds: across inputs.
    with pytest.raises(TypeError, match="one kind and unit"):
        list(p(zip(v, t_s), zip(v, t_ns)))
    with pytest.raises(TypeError, match="one kind and unit"):
        list(p(zip(v, t_s), zip(v, range(3))))
    # Live: across inputs and in advance().
    live = p.live().push("a", np.datetime64(1, "s"), 1.0)
    with pytest.raises(TypeError, match="one kind and unit"):
        live.push("b", np.datetime64(1, "ns"), 1.0)
    with pytest.raises(TypeError, match="one kind and unit"):
        live.push("b", 1, 1.0)
    with pytest.raises(TypeError, match="one kind and unit"):
        live.advance(np.datetime64(2, "ns"))
    with pytest.raises(TypeError, match="one kind and unit"):
        live.advance(2)
    live.push("b", np.datetime64(2, "s"), 1.0).advance(np.datetime64(3, "s"))
    # Plain numbers lock too: a time after an int raises.
    with pytest.raises(TypeError, match="index is a datetime64"):
        p.live().push("a", 0, 1.0).push("a", np.datetime64(1, "s"), 1.0)
    # Merge iterator feeds share one kind.
    with pytest.raises(TypeError, match="one kind and unit"):
        list(Merge()(zip(v, t_s), zip(v, t_ns)))


def test_rejected_push_does_not_lock_the_index_kind():
    p = _pipe()
    live = p.live()
    for bad in ("abc", 1 + 2j, [[1.0, 2.0]]):
        with pytest.raises((TypeError, ValueError)):
            live.push("a", np.datetime64(1, "s"), bad)
    with pytest.raises(TypeError, match="finite"):
        live.advance(np.datetime64("NaT", "ms"))
    live.push("a", np.datetime64(5, "ns"), 1.0)      # the first taken event sets ns
    with pytest.raises(TypeError, match="one kind and unit"):
        live.push("a", np.datetime64(6, "s"), 1.0)
    np.testing.assert_array_equal(live.result()[1], [5])


def test_time_index_non_native_byte_order():
    p = _pipe()
    _, idx = p((np.array([1.0, 2.0]), np.array([1, 2], dtype=">M8[ns]")))
    np.testing.assert_array_equal(idx, [1, 2])


def test_iterator_none_is_nan_and_matches_batch():
    p = _pipe()
    vals = [1.0, None, 3.0, 4.0]
    lazy = list(p(iter([(v, i) for i, v in enumerate(vals)])))
    bv, _ = p((np.array([1.0, np.nan, 3.0, 4.0]), np.arange(4)))
    np.testing.assert_array_equal([e[0] for e in lazy], bv)
    wide = list(p(iter([([1.0], 0), ([None], 1)])))
    assert np.isnan(wide[1][0])
    out = list(Merge()(iter([(None, 1)]), iter([(2.0, 2)])))
    assert np.isnan(out[0][0]) and out[1][0] == 2.0


@pytest.mark.parametrize("empty", [[], (), np.array([])])
def test_iterator_empty_event_raises(empty):
    p = _pipe()
    with pytest.raises(ValueError, match="the value of event 1 of input 0 is empty"):
        list(p(iter([(1.0, 0), (empty, 1)])))


@pytest.mark.parametrize("text", [b"3", bytearray(b"3")])
def test_iterator_text_value_raises(text):
    p = _pipe()
    with pytest.raises(TypeError, match="the value of event 0 of input 0"):
        list(p(iter([(text, 0)])))


@pytest.mark.parametrize("text", ["abc", "3", "", "nan"])
def test_any_string_value_raises_typeerror(text):
    p = _pipe()
    err = r"str|string"
    for call in (lambda: p([1.0, text]),
                 lambda: p((np.array([1.0, text], dtype=object), np.arange(2))),
                 lambda: p(pd.Series([1.0, text], dtype=object)),
                 lambda: p.live().push("a", 0, text),
                 lambda: p.live().push("a", 0, [1.0, text]),
                 lambda: p.live().push("a", 0, np.str_(text)),
                 lambda: list(p(iter([(text, 0)]))),
                 lambda: s.RollingMean(2)(text),
                 lambda: s.RollingMean(2)([1.0, text]),
                 lambda: s.RollingMean(2)(np.array([1.0, text], dtype=object)),
                 lambda: s.RollingMean(2)(np.array(["1", text])),
                 lambda: list(s.RollingMean(2)(iter([1.0, text]))),
                 lambda: list(s.RollingMean(2)(iter([np.array([1.0]), [text]]))),
                 lambda: s.RollingCorr(3)([1.0, text], [1.0, 2.0]),
                 lambda: merge((np.array(["1", text]), np.array([0, 1]))),
                 lambda: combine_latest((np.array(["1", text]), np.array([0, 1])),
                                        (np.array([1.0, 2.0]), np.array([0, 1]))),
                 lambda: dropna(["1", text]),
                 lambda: resample(["1", text], every=2)):
        with pytest.raises(TypeError, match=err):
            call()
    # A ragged sequence keeps its shape error.
    with pytest.raises(ValueError):
        p.live().push("a", 0, [[1.0], [1.0, 2.0]])


def test_string_settings_raise_typeerror():
    for call in (lambda: resample([1.0, 2.0], threshold="3"),
                 lambda: resample([1.0, 2.0], every="2"),
                 lambda: resample([1.0, 2.0], count="2"),
                 lambda: s.Delay("1"),
                 lambda: s.Select("0"),
                 lambda: Pipeline([Input("a")], [Input("a")], max_pending="5")):
        with pytest.raises(TypeError, match="must be a number"):
            call()


def test_none_is_nan_on_every_operator_path():
    import asyncio
    rm = lambda: s.RollingMean(2)
    nan = np.nan
    assert np.isnan(rm()(None))
    xs = [1.0, 2.0, None, 4.0, 6.0]
    ref = rm()(np.array([1.0, 2.0, nan, 4.0, 6.0]))
    op = rm()
    np.testing.assert_array_equal([op(x) for x in xs], ref)
    np.testing.assert_array_equal(rm()(xs), ref)
    np.testing.assert_array_equal(list(rm()(iter(xs))), ref)

    async def agen():
        for x in xs:
            yield x

    async def collect():
        return [x async for x in rm()(agen())]
    np.testing.assert_array_equal(asyncio.run(collect()), ref)
    np.testing.assert_array_equal(s.Add()(None, 1.0), nan)
    np.testing.assert_array_equal(s.Add()([1.0, None], [1.0, 1.0]), [2.0, nan])
    np.testing.assert_array_equal(s.Add()([(1.0, None), (2.0, 3.0)]), [nan, 5.0])


def test_resample_clock_fractional_index_raises():
    v = (np.arange(3.0), np.array([0.0, 1.5, 2.0]))
    c = (np.arange(2.0), np.array([1, 3]))
    with pytest.raises(TypeError, match="integer-valued"):
        resample(v, c, clock=True)
    with pytest.raises(TypeError, match="integer-valued"):
        resample((np.arange(3.0), np.array([0, 1, 2])), (np.arange(2.0), np.array([1.5, 3])),
                 clock=True)
    vals, idx = resample((np.arange(3.0), np.array([0.0, 1.0, 2.0])), c, clock=True)
    np.testing.assert_array_equal(idx, [1, 3])
    np.testing.assert_array_equal(vals, [1.0, 2.0])
