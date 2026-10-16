import importlib.util
import pathlib

import numpy as np
import pytest
from screamer import TripleBarrier, TripleBarrierOHLC, BacktestOHLCTarget

_B = pathlib.Path(__file__).resolve().parents[1] / "devtools" / "baselines"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, _B / f"{name}.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return getattr(m, f"{name}_numpy")


TB_ref, TBO_ref = _load("TripleBarrier"), _load("TripleBarrierOHLC")


def _walk(n, seed, vol=0.01):
    rng = np.random.default_rng(seed)
    return 100 * np.exp(np.cumsum(rng.standard_normal(n) * vol))


def _bars(n, seed):
    rng = np.random.default_rng(seed)
    c = _walk(n, seed)
    o = np.r_[c[0], c[:-1]] * np.exp(rng.standard_normal(n) * 0.003)
    hi = np.maximum(o, c) * np.exp(np.abs(rng.standard_normal(n)) * 0.004)
    lo = np.minimum(o, c) * np.exp(-np.abs(rng.standard_normal(n)) * 0.004)
    return o, hi, lo, c


def test_hand_example():
    p = np.array([100, 101, 103, 99, 100, 100, 97, 100.0])
    out = np.asarray(TripleBarrier(horizon=3, upper=0.02, lower=0.02)(p))
    assert np.isnan(out[:3]).all()
    np.testing.assert_allclose(out[3], [1.0, 0.03, 2.0])             # entry 0 (100): U 102, 103 at k=2
    np.testing.assert_allclose(out[4], [0.0, 100 / 101 - 1, 3.0])     # entry 1 (101): 103, 99, 100 inside (98.98, 103.02)
    np.testing.assert_allclose(out[5], [-1.0, 99 / 103 - 1, 1.0])     # entry 2 (103): L 100.94, 99 at k=1
    np.testing.assert_allclose(out[6], [-1.0, 97 / 99 - 1, 3.0])      # entry 3 (99): L 97.02, 97 at k=3
    np.testing.assert_allclose(out[7], [-1.0, 97 / 100 - 1, 2.0])     # entry 4 (100): L 98, 97 at k=2


@pytest.mark.parametrize("unit,up,lo", [("fraction", 0.02, 0.01), ("price", 1.5, 2.0),
                                        ("vol", 2.0, 1.0), ("fraction", 0.0, 0.03),
                                        ("fraction", 0.0, 0.0)])
@pytest.mark.parametrize("h", [1, 5, 20])
def test_matches_baseline(unit, up, lo, h):
    p = _walk(400, h)
    got = np.asarray(TripleBarrier(horizon=h, upper=up, lower=lo, barrier_unit=unit, vol_span=30)(p))
    want = TB_ref(h, up, lo, unit, 30)(p)
    np.testing.assert_allclose(got, want, rtol=1e-12, atol=1e-12, equal_nan=True)


@pytest.mark.parametrize("unit", ["fraction", "price", "vol"])
def test_ohlc_matches_baseline(unit):
    o, hi, lo, c = _bars(400, 7)
    args = dict(horizon=10, upper=0.01 if unit != "price" else 1.0,
                lower=0.01 if unit != "price" else 1.0, barrier_unit=unit, vol_span=30)
    if unit == "vol":
        args.update(upper=1.5, lower=1.5)
    got = np.asarray(TripleBarrierOHLC(**args)(o, hi, lo, c))
    want = TBO_ref(*args.values())(o, hi, lo, c)
    np.testing.assert_allclose(got, want, rtol=1e-12, atol=1e-12, equal_nan=True)


def test_nan_and_nonpositive_prices_only_hit_their_window():
    p = _walk(200, 3)
    p[50] = np.nan
    p[120] = -1.0
    got = np.asarray(TripleBarrier(horizon=5, upper=0.01, lower=0.01)(p))
    want = TB_ref(5, 0.01, 0.01, "fraction", 100)(p)
    np.testing.assert_allclose(got, want, equal_nan=True)
    assert np.isfinite(got[100:115]).all()                       # untouched entries stay valid


def test_vol_unit_skips_bad_prices_in_the_volatility():
    p = _walk(300, 4)
    p[80] = np.nan
    p[150] = 0.0
    got = np.asarray(TripleBarrier(horizon=5, upper=1.0, lower=1.0, barrier_unit="vol",
                                   vol_span=20)(p))
    want = TB_ref(5, 1.0, 1.0, "vol", 20)(p)
    np.testing.assert_allclose(got, want, rtol=1e-12, atol=1e-12, equal_nan=True)
    assert np.isfinite(got[-50:]).all()                          # the volatility recovers


def test_vol_unit_on_constant_price_is_nan():
    out = np.asarray(TripleBarrier(horizon=3, upper=1.0, lower=1.0, barrier_unit="vol")(np.full(50, 100.0)))
    assert np.isnan(out).all()


def test_ohlc_gap_at_open_and_both_inside():
    # entry close 100; bar 1 opens at 103 (gap through U = 102): exit at the open
    o = np.array([100, 103, 100.0]); h = np.array([100, 104, 100.0])
    l = np.array([100, 102.5, 100.0]); c = np.array([100, 103, 100.0])
    out = np.asarray(TripleBarrierOHLC(horizon=1, upper=0.02, lower=0.02)(o, h, l, c))
    np.testing.assert_allclose(out[1], [1.0, 0.03, 1.0])
    # bar holding both barriers (high 103, low 97): unknown order, NaN
    o2 = np.array([100, 100, 100.0]); h2 = np.array([100, 103, 100.0])
    l2 = np.array([100, 97, 100.0]); c2 = np.array([100, 100, 100.0])
    out2 = np.asarray(TripleBarrierOHLC(horizon=1, upper=0.02, lower=0.02)(o2, h2, l2, c2))
    assert np.isnan(out2[1]).all()


def test_ohlc_exit_is_the_barrier_inside_the_bar():
    # bar 1 opens inside, its high reaches U = 102: exit at the level, not the high
    o = np.array([100, 100.5, 100.0]); h = np.array([100, 104, 100.0])
    l = np.array([100, 100, 100.0]); c = np.array([100, 103, 100.0])
    out = np.asarray(TripleBarrierOHLC(horizon=2, upper=0.02, lower=0.02)(o, h, l, c))
    np.testing.assert_allclose(out[2], [1.0, 0.02, 1.0])


def test_ohlc_exits_agree_with_backtest_engine():
    # a long entered at close[e] with stop and take of 1%: the engine's exit
    # bar equals the label's exit bar when no bar is ambiguous
    o, hi, lo, c = _bars(300, 11)
    lab = np.asarray(TripleBarrierOHLC(horizon=10, upper=0.01, lower=0.01)(o, hi, lo, c))
    checked = 0
    for e in range(0, 280, 7):
        row = lab[e + 10]
        if not np.isfinite(row[0]) or row[0] == 0:
            continue
        tgt = np.zeros(300); tgt[e] = 1.0; tgt[e + 1:e + 11] = 1.0
        bt = np.asarray(BacktestOHLCTarget(fill_at="close", delay=0, stop_loss=0.01,
                                           take_profit=0.01)(tgt, o, hi, lo, c))
        k = int(row[2])
        pos = bt[:, 2]
        assert pos[e + k] == 0.0 and pos[e + k - 1] == 1.0       # exited on bar e + k
        np.testing.assert_allclose(bt[e + k, 0] - bt[e - 1, 0] if e else bt[k, 0],
                                   row[1] * c[e], rtol=1e-9)      # at the same price
        checked += 1
    assert checked >= 5


def test_no_lookahead():
    p = _walk(300, 5)
    a = np.asarray(TripleBarrier(horizon=8, upper=0.01, lower=0.01)(p))
    q = p.copy(); q[200:] *= 1.5
    b = np.asarray(TripleBarrier(horizon=8, upper=0.01, lower=0.01)(q))
    np.testing.assert_array_equal(a[:200], b[:200])


def test_many_overlapping_entries():
    p = _walk(1100, 9)
    got = np.asarray(TripleBarrier(horizon=50, upper=0.005, lower=0.005)(p))
    want = TB_ref(50, 0.005, 0.005, "fraction", 100)(p)
    np.testing.assert_allclose(got, want, equal_nan=True)


def test_stream_equals_batch_and_reset():
    o, hi, lo, c = _bars(200, 13)
    tb = TripleBarrierOHLC(horizon=6, upper=1.0, lower=1.0, barrier_unit="vol", vol_span=20)
    batch = np.asarray(tb(o, hi, lo, c))
    tb.reset()
    stream = np.array([np.asarray(tb(*(float(a[i]) for a in (o, hi, lo, c)))).ravel()
                       for i in range(200)])
    np.testing.assert_array_equal(stream, batch)


def test_panel_equals_columns():
    P = np.column_stack([_walk(200, s) for s in range(4)])
    tb = TripleBarrier(horizon=5, upper=0.01, lower=0.01)
    whole = np.asarray(tb(P))
    assert whole.shape == (200, 4, 3)
    for k in range(4):
        np.testing.assert_array_equal(whole[:, k, :],
                                      np.asarray(TripleBarrier(horizon=5, upper=0.01, lower=0.01)(P[:, k])))


def test_settings_are_checked():
    with pytest.raises(ValueError):
        TripleBarrier(horizon=0)
    with pytest.raises(ValueError):
        TripleBarrier(upper=-0.1)
    with pytest.raises(ValueError):
        TripleBarrier(upper=np.inf)
    with pytest.raises(ValueError):
        TripleBarrier(lower=1.0)                                   # fraction below 1
    TripleBarrier(lower=1.0, barrier_unit="price")                 # fine in price units
    with pytest.raises(ValueError):
        TripleBarrier(vol_span=0.5)
    with pytest.raises(ValueError, match="fraction.*price.*vol"):
        TripleBarrier(barrier_unit="ticks")
    with pytest.raises(ValueError, match="fraction.*price.*vol"):
        TripleBarrierOHLC(barrier_unit="ticks")


def test_pairs_with_forecast_pairs_at_the_entry():
    from screamer import forecast_pairs
    p = _walk(100, 2)
    y = np.asarray(TripleBarrier(horizon=4, upper=0.01, lower=0.01)(p))[:, 0]
    _, ys = forecast_pairs(p, y, count=4)
    want = TB_ref(4, 0.01, 0.01, "fraction", 100)(p)[4:, 0]
    np.testing.assert_array_equal(ys[:96], want)                  # row e holds entry e's label


@pytest.mark.parametrize("ohlc", [False, True])
def test_regimes_agree(ohlc):
    # eager, graph (Pipeline), and lazy (event iterator) give the same rows
    import screamer
    o, hi, lo, c = _bars(150, 17)
    arrays = [o, hi, lo, c] if ohlc else [c]
    op = (lambda: TripleBarrierOHLC(horizon=6, upper=1.0, lower=1.0, barrier_unit="vol",
                                    vol_span=20)) if ohlc else \
         (lambda: TripleBarrier(horizon=6, upper=0.01, lower=0.01))
    eager = np.asarray(op()(*arrays))
    index = np.arange(150)
    nodes = [screamer.Input(f"in{i}") for i in range(len(arrays))]
    source = nodes[0] if not ohlc else screamer.CombineLatest()(*nodes)
    pipeline = screamer.Pipeline(inputs=nodes, outputs=[op()(source)])
    graph, _ = pipeline(*[(a, index) for a in arrays])
    np.testing.assert_array_equal(np.asarray(graph), eager)
    lazy = list(pipeline(*[((float(v), int(i)) for i, v in enumerate(a)) for a in arrays]))
    np.testing.assert_array_equal(np.array([np.asarray(r[0]).ravel() for r in lazy]), eager)
