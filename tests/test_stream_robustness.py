"""Stream robustness (Task 30): a decreasing index raises at every stream entry
point (G26), a fill Resample bounds the empty bars of one gap (G22), and a
clock Resample honors watermarks, so its bars do not depend on the drain or
advance() timing and advance(now) releases aligned rows while the clock is
silent (G19, G20). The JavaScript twin is js/test/stream_robustness.test.ts."""
import resource
import sys

import numpy as np
import pytest

import screamer as sc
from screamer import Input, Pipeline
from screamer.streams import merge
from screamer.supervised import forecast_pairs

DEC = r"the index of input 0 decreases at event 2, from 20 to 10; a stream's index must be non-decreasing"
V = np.array([1.0, 2.0, 3.0, 4.0])
K = np.array([0, 20, 10, 30])           # decreases at event 2
EQ = np.array([0, 10, 10, 30])          # equal indexes are allowed
W, KW = np.array([5.0, 6.0]), np.array([1, 2])


def gen(v, k):
    return iter([(float(a), int(b)) for a, b in zip(v, k)])


def pipe(op=lambda x: sc.RollingMean(1)(x)):
    x = Input("x")
    return Pipeline([x], [op(x)])


def pipe2():
    a, b = Input("a"), Input("b")
    return Pipeline([a, b], [sc.RollingMean(1)(a), sc.RollingMean(1)(b)])


def live(ks, p=None):
    p = p or pipe()
    s = p.live()
    for val, k in zip(V, ks):
        s.push("x", int(k), float(val))
    s.flush()
    return s.result()


def live_wide(ks):
    x = Input("x")
    s = Pipeline([x], [sc.Select(0)(x)]).live()
    for val, k in zip(V, ks):
        s.push("x", int(k), [float(val), float(val)])
    s.flush()
    return s.result()


def clock_resample(k):
    return sc.Resample((V, k), (np.ones(2), np.array([15, 40])), clock=True, agg="last")


ENTRY_POINTS = {
    "merge": lambda k: merge((V, k), (W, KW)),
    "merge lazy": lambda k: list(merge(gen(V, k), gen(W, KW))),
    "Merge": lambda k: sc.Merge()((V, k), (W, KW)),
    "CombineLatest": lambda k: sc.CombineLatest()((V, k), (W, KW)),
    "CombineLatest lazy": lambda k: list(sc.CombineLatest()(gen(V, k), gen(W, KW))),
    "Pipeline": lambda k: pipe()((V, k)),
    "Pipeline 2 inputs": lambda k: pipe2()((V, k), (W, KW)),
    "Pipeline lazy": lambda k: list(pipe()(gen(V, k))),
    "Pipeline live": live,
    "Pipeline live wide": live_wide,
    "Resample every": lambda k: sc.Resample(freq=10, agg="last")((V, k)),
    "Resample count": lambda k: sc.Resample(count=2, agg="last")((V, k)),
    "Resample lazy": lambda k: list(sc.Resample(freq=10, agg="last")(gen(V, k))),
    "Resample clock": clock_resample,
    "Delay": lambda k: sc.Delay(5)(V, k),
    "Dropna": lambda k: sc.Dropna()((V, k)),
    "Select": lambda k: sc.Select(0)((np.c_[V, V], k)),
    "Filter": lambda k: sc.Filter()((V, k), (np.ones(4), k)),
    "forecast_pairs": lambda k: forecast_pairs((V, k), (V, k), duration=5),
    "forecast_pairs lazy": lambda k: list(forecast_pairs(gen(V, k), gen(V, k), duration=5)),
}


@pytest.mark.parametrize("name", list(ENTRY_POINTS))
def test_a_decreasing_index_raises_naming_the_input_and_event(name):
    with pytest.raises(ValueError, match=DEC):
        ENTRY_POINTS[name](K)


@pytest.mark.parametrize("name", list(ENTRY_POINTS))
def test_equal_indexes_stay_allowed(name):
    ENTRY_POINTS[name](EQ)


def test_split_names_its_stream():
    msg = r"the index of the split stream decreases at event 2, from 20 to 10"
    with pytest.raises(ValueError, match=msg):
        sc.split((V, K), [0, 1, 0, 1])
    with pytest.raises(ValueError, match=r"from 2 to 1\.5"):
        sc.split((V, np.array([0.5, 2.0, 1.5, 3.0])), [0, 1, 0, 1])
    with pytest.raises(ValueError, match=msg):
        sc.split((np.c_[V, V], K), [0, 1, 0, 1])
    sc.split((V, EQ), [0, 1, 0, 1])
    sc.split(V, [0, 1, 0, 1])                       # positional: no index to check


def test_the_second_input_is_named():
    with pytest.raises(ValueError, match=r"the index of input 1 decreases at event 1, from 2 to 1"):
        pipe2()((V, EQ), (W, np.array([2, 1])))
    with pytest.raises(ValueError, match=r"the index of input 1 decreases at event 1, from 2 to 1"):
        merge((V, EQ), (W, np.array([2, 1])))


def test_a_rejected_live_event_leaves_the_session_usable():
    p = pipe()
    s = p.live()
    s.push("x", 0, 1.0).push("x", 20, 2.0)
    with pytest.raises(ValueError, match="decreases at event 2, from 20 to 10"):
        s.push("x", 10, 3.0)
    s.push("x", 30, 4.0).flush()
    v, k = s.result()
    want = p((np.array([1.0, 2.0, 4.0]), np.array([0, 20, 30])))
    np.testing.assert_array_equal(k, want[1])
    np.testing.assert_array_equal(v, want[0])


def test_a_live_event_below_an_advance_raises():
    s = pipe().live()
    s.push("x", 0, 1.0).advance(50)
    with pytest.raises(ValueError, match=r"the index of input 0 is 40 at event 1, below an earlier advance\(50\)"):
        s.push("x", 40, 2.0)
    s.push("x", 50, 2.0)                    # at the advanced time: allowed
    s2 = pipe2().live().push("a", 0, 1.0).advance(50)
    s2.advance(20)                          # a lower advance does not lower the bound
    with pytest.raises(ValueError, match=r"the index of input 1 is 45 at event 0, below an earlier advance\(50\)"):
        s2.push("b", 45, 2.0)
    pipe().live().push("x", 0, 1.0)         # a new session starts afresh


# ---------------------------------------------------------------------------
# G22: the fill limit
# ---------------------------------------------------------------------------

FILL = r"a fill Resample would emit {} empty bars for one gap, more than max_fill={}"


@pytest.mark.parametrize("fill", ["nan", "carry"])
def test_a_huge_gap_raises_before_it_fills(fill):
    gap = np.array([0, 2_000_001])
    with pytest.raises(ValueError, match=FILL.format(2_000_000, 1_000_000)):
        sc.Resample(freq=1, fill=fill)((V[:2], gap))
    v, k = sc.Resample(freq=1, fill=fill, max_fill=2_000_000)((V[:2], gap))
    assert len(k) == 2_000_002
    v, k = sc.Resample(freq=10, fill=fill)((V[:2], np.array([0, 95])))     # a normal gap
    np.testing.assert_array_equal(k, np.arange(0, 100, 10))


def test_the_limit_counts_the_empty_bars_of_one_gap():
    # 5 empty buckets between bars 0 and 60: max_fill=5 passes, 4 raises, in
    # batch, lazy and live, through the builtin and a functor reducer.
    feed = (V[:2], np.array([0, 60]))
    for agg in ("last", sc.Last()):
        _, k = sc.Resample(freq=10, fill="nan", max_fill=5, agg=agg)(feed)
        np.testing.assert_array_equal(k, np.arange(0, 70, 10))
        with pytest.raises(ValueError, match=FILL.format(5, 4)):
            sc.Resample(freq=10, fill="nan", max_fill=4, agg=agg)(feed)
        with pytest.raises(ValueError, match=FILL.format(5, 4)):
            list(sc.Resample(freq=10, fill="nan", max_fill=4, agg=agg)(gen(*feed)))
        x = Input("x")
        p = Pipeline([x], [sc.Resample(freq=10, fill="nan", max_fill=4, agg=agg)(x)])
        with pytest.raises(ValueError, match=FILL.format(5, 4)):
            p.live().push("x", 0, 1.0).push("x", 60, 2.0)
    # Skip fills nothing, so no limit applies.
    _, k = sc.Resample(freq=10, max_fill=1)(feed)
    np.testing.assert_array_equal(k, [0, 60])


def test_a_huge_advance_raises_cleanly():
    x = Input("x")
    p = Pipeline([x], [sc.Resample(freq=10, fill="nan")(x)])
    s = p.live().push("x", 0, 1.0)
    with pytest.raises(ValueError, match=r"a fill Resample would emit 900719925474098 empty bars"):
        s.advance(2**53)
    s = p.live().push("x", 0, 1.0).advance(1000)          # 99 empty bars: fine
    v, k = s.flush().result()
    np.testing.assert_array_equal(k, np.arange(0, 1000, 10))
    s = p.live().push("x", 0, 1.0).advance(2**63 - 1)     # the end of input fills nothing
    v, k = s.result()
    np.testing.assert_array_equal(k, [0])


def test_a_rejected_advance_leaves_the_session_usable():
    # The resample checks the gap before it changes anything, so the session
    # goes on, as after a rejected event; its time has moved to the rejected
    # advance (no later event may come below it).
    x = Input("x")
    p = Pipeline([x], [sc.RollingMean(1)(x), sc.Resample(freq=10, fill="nan", max_fill=50)(x)],
                 align_outputs=False)
    s = p.live().push("x", 0, 1.0).push("x", 5, 2.0)
    with pytest.raises(ValueError, match=FILL.format(99, 50)):
        s.advance(1000)
    with pytest.raises(ValueError, match=r"below an earlier advance\(1000\)"):
        s.push("x", 999, 3.0)
    s.advance(300).advance(650)                        # gaps of 29 and 35 bars: fine
    s.push("x", 1000, 3.0).flush()
    (_, k0), (_, k1) = s.result()
    np.testing.assert_array_equal(k0, [0, 5, 1000])
    np.testing.assert_array_equal(k1, np.arange(0, 1010, 10))


def test_the_fill_limit_counts_one_step():
    # The limit counts the empty bars that one event or one advance() closes.
    # Batch closes the gap 0 to 300 in one step and raises; a live session that
    # advances in steps of 30 fills the same gap (the documented per-step rule).
    x = Input("x")
    p = Pipeline([x], [sc.Resample(freq=10, fill="nan", max_fill=4)(x)])
    with pytest.raises(ValueError, match=FILL.format(29, 4)):
        p((np.array([1.0, 2.0]), np.array([0, 300])))
    s = p.live().push("x", 0, 1.0)
    for now in range(30, 301, 30):
        s.advance(now)
    s.push("x", 300, 2.0).flush()
    _, k = s.result()
    np.testing.assert_array_equal(k, np.arange(0, 310, 10))


def test_live_pushes_out_of_order_across_inputs_match_batch():
    # The reviewer's repro: each input in order, the inputs ahead of each
    # other. The value at 100 must not join the bar at 60 (lookahead).
    a, c = Input("a"), Input("c")
    feeds = ((np.array([1.0, 2.0]), np.array([50, 100])), (np.ones(2), np.array([60, 120])))
    p = Pipeline([a, c], [sc.Resample(a, c, clock=True, agg="last")])
    v, k = p(*feeds)
    np.testing.assert_array_equal(k, [60, 120])
    np.testing.assert_array_equal(v, [1.0, 2.0])
    s = p.live()
    s.push("a", 50, 1.0).push("a", 100, 2.0).push("c", 60, 1.0).push("c", 120, 1.0).flush()
    v2, k2 = s.result()
    np.testing.assert_array_equal(k2, k)
    np.testing.assert_array_equal(v2, v)
    # With an aligned output beside it and an advance, no row settles early.
    p2 = Pipeline([a, c], [sc.RollingMean(1)(a), sc.Resample(a, c, clock=True, agg="last")])
    want = p2(*feeds)
    s = p2.live()
    s.push("a", 50, 1.0).push("a", 100, 2.0).advance(55)
    r1 = s.result()
    s.push("c", 60, 1.0).push("c", 120, 1.0).flush()
    r2 = s.result()
    for j in range(2):
        np.testing.assert_array_equal(np.concatenate([r1[j][1], r2[j][1]]), want[j][1])
        np.testing.assert_array_equal(np.concatenate([r1[j][0], r2[j][0]]), want[j][0])


def test_max_fill_follows_the_count_setting_rule():
    for bad in (0, -1, 2**53 + 1, 2**64):
        with pytest.raises(ValueError, match="max_fill must be at least 1 and at most 2"):
            sc.Resample(freq=10, fill="nan", max_fill=bad)
    for bad in (1.5, True, "3", None.__class__, float("nan")):
        with pytest.raises(TypeError, match="max_fill must be an integer"):
            sc.Resample(freq=10, fill="nan", max_fill=bad)
    for ok in (1, 2**53, np.int64(7), 7.0, np.float32(7.0)):
        sc.Resample(freq=10, fill="nan", max_fill=ok)
    from screamer import screamer_bindings as _b
    assert _b._DEFAULT_MAX_FILL == 1_000_000


def test_max_fill_round_trips_through_to_dict():
    x = Input("x")
    p = Pipeline([x], [sc.Resample(freq=10, fill="nan", max_fill=4)(x)])
    d = p.to_dict()
    assert '"max_fill": 4' in p.to_json()
    q = Pipeline.from_dict(d)
    with pytest.raises(ValueError, match=FILL.format(5, 4)):
        q((V[:2], np.array([0, 60])))
    plain = Pipeline([x], [sc.Resample(freq=10, fill="nan")(x)])
    assert "max_fill" not in plain.to_json()


# ---------------------------------------------------------------------------
# G19, G20: clock resamples honor watermarks
# ---------------------------------------------------------------------------

def _clock_via_filter():
    x, m, c = Input("x"), Input("m"), Input("c")
    node = sc.Resample(sc.Filter()(x, m), c, clock=True, agg="last")
    feeds = ((np.array([1.0]), np.array([5])), (np.ones(3), np.array([0, 5, 20])),
             (np.zeros(3), np.array([6, 9, 12])))
    return Pipeline([x, m, c], [node]), feeds


def test_a_gated_value_joins_the_bar_of_its_tick():
    # x at 5 passes the Filter only once the mask moves past 5 (at 20), after
    # the ticks at 6, 9 and 12; it belongs to the bar at 6 in every mode.
    p, feeds = _clock_via_filter()
    v, k = p(*feeds)
    np.testing.assert_array_equal(k, [6])
    np.testing.assert_array_equal(v, [1.0])
    assert list(p(*(gen(*f) for f in feeds))) == [(1.0, 6)]
    for adv in (None, 6, 8, 11, 19):
        s = p.live()
        events = sorted((int(i), n, float(val)) for n, (vals, idx) in zip("xmc", feeds)
                        for val, i in zip(vals, idx))
        parts = []
        for i, n, val in events:
            if adv is not None and i > adv and not parts:
                s.advance(adv)
                parts.append(s.result())
            s.push(n, i, val)
            parts.append(s.result())
        s.flush()
        parts.append(s.result())
        got = np.concatenate([q[1] for q in parts])
        np.testing.assert_array_equal(got, [6], err_msg=f"advance({adv})")


def test_advance_releases_aligned_rows_while_the_clock_is_silent():
    a, c = Input("a"), Input("c")
    p = Pipeline([a, c], [sc.RollingMean(1)(a), sc.Resample(a, c, clock=True, agg="last",
                                                            fill="carry")])
    va, ka = np.arange(1.0, 50.0), np.arange(1, 50)
    want = p((va, ka), (np.ones(2), np.array([2, 60])))
    s = p.live()
    s.push("a", 1, 1.0).push("c", 2, 1.0)
    for t in range(2, 50):
        s.push("a", t, float(t))
    (_, k0), _ = s.result()
    assert len(k0) <= 1                          # rows wait for the silent clock
    s.advance(50)
    (_, k1), _ = s.result()
    np.testing.assert_array_equal(np.concatenate([k0, k1]), want[0][1][want[0][1] < 50])
    s.push("c", 60, 1.0).flush()
    (_, k2), _ = s.result()
    np.testing.assert_array_equal(np.concatenate([k0, k1, k2]), want[0][1])


def _maxrss_mb():
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / 2**20 if sys.platform == "darwin" else r / 2**10


def test_a_stalled_clock_with_periodic_advance_keeps_memory_flat():
    a, c = Input("a"), Input("c")
    p = Pipeline([a, c], [sc.RollingMean(5)(a), sc.Resample(a, c, clock=True, agg="last")])

    def run(n, s):
        for t in range(n):
            s.push("a", t, float(t))
            if t % 100 == 99:
                s.advance(t)
                s.result()

    s = p.live().push("c", 0, 1.0)
    run(200_000, s)                               # warm-up: the allocator's high water mark
    before = _maxrss_mb()
    s = p.live().push("c", 0, 1.0)
    run(1_000_000, s)
    assert _maxrss_mb() - before < 8
