import sys

import numpy as np
import pytest

import screamer as s
from screamer.screamer_bindings import _circular_shift, _segment_shuffle

X = np.array([np.nan, np.nan, 1, 1, 1, 0, 0, -1, -1, -1, -1, 2, 0, 0, 1.0])


def _runs(v):
    """(value, length) runs of a 1-D array without NaN."""
    out, i = [], 0
    while i < len(v):
        j = i + 1
        while j < len(v) and v[j] == v[i]:
            j += 1
        out.append((v[i], j - i))
        i = j
    return out


# --- surrogate generators -------------------------------------------------

def test_golden_values():
    # Fixed outputs for a fixed seed. The JS/WASM tests check the same values,
    # so a change here is a cross-platform determinism break.
    np.testing.assert_array_equal(
        s.circular_shift(X, 2, min_shift=2, seed=42)[:, 0],
        [np.nan, np.nan, 2, 0, 0, 1, 1, 1, 1, 0, 0, -1, -1, -1, -1])
    np.testing.assert_array_equal(
        s.segment_shuffle(X, 2, seed=42)[:, 1],
        [np.nan, np.nan, 0, 0, 1, 1, 1, -1, -1, -1, -1, 2, 0, 0, 1])


def test_circular_shift_is_a_rotation_within_bounds():
    rng = np.random.default_rng(0)
    x = np.r_[np.full(5, np.nan), np.round(rng.standard_normal(200))]
    out = s.circular_shift(x, 300, min_shift=10, seed=3)
    assert out.shape == (205, 300)
    valid = x[5:]
    L = len(valid)
    for j in range(out.shape[1]):
        col = out[:, j]
        assert np.isnan(col[:5]).all()
        rotations = [k for k in range(L) if np.array_equal(np.roll(valid, k), col[5:])]
        assert any(10 <= k <= L - 10 for k in rotations)


def test_segment_shuffle_keeps_the_runs():
    # Distinct run values so neighbouring runs can never merge.
    lengths = [3, 1, 4, 1, 5, 9, 2, 6]
    x = np.r_[np.nan, np.repeat(np.arange(len(lengths), dtype=float), lengths)]
    out = s.segment_shuffle(x, 50, seed=7)
    want = sorted(zip(np.arange(len(lengths), dtype=float), lengths))
    orders = set()
    for j in range(out.shape[1]):
        assert np.isnan(out[0, j])
        got = _runs(out[1:, j])
        assert sorted(got) == want
        orders.add(tuple(v for v, _ in got))
    assert len(orders) > 40                       # the order really changes


@pytest.mark.parametrize("gen", ["shift", "segments"])
def test_seeded_and_chunk_invariant(gen):
    rng = np.random.default_rng(1)
    x = np.round(rng.standard_normal(120))
    def make(col0, k, seed=5):
        if gen == "shift":
            return _circular_shift(x, 3, seed, col0, k)
        return _segment_shuffle(x, seed, col0, k)
    full = make(0, 40)
    np.testing.assert_array_equal(full, make(0, 40))
    np.testing.assert_array_equal(full, np.hstack([make(0, 7), make(7, 20), make(27, 13)]))
    assert not np.array_equal(full, make(0, 40, seed=6))


def test_shift_argument_checks():
    with pytest.raises(ValueError):
        s.circular_shift(X, 3, min_shift=0)
    with pytest.raises(ValueError):
        s.circular_shift(X, 3, min_shift=7)       # more than half of L = 13
    with pytest.raises(ValueError):
        s.circular_shift(np.array([np.nan, 1.0]), 3)


def test_read_only_and_list_inputs():
    ro = X.copy(); ro.flags.writeable = False
    np.testing.assert_array_equal(s.segment_shuffle(ro, 3, seed=1),
                                  s.segment_shuffle(list(X), 3, seed=1))


# --- position_test ----------------------------------------------------------

def _market(T, seed):
    rng = np.random.default_rng(seed)
    ret = rng.standard_normal(T) * 0.01
    c = 100 * np.exp(np.cumsum(ret))
    o = np.r_[c[0], c[:-1]]
    return ret, o, np.maximum(o, c) * 1.001, np.minimum(o, c) * 0.999, c


def _held(sign, hold):
    """Hold each decision for `hold` bars, so the signal has real trades."""
    return np.repeat(sign[::hold], hold)[:len(sign)]


def test_observed_matches_backtest_report():
    ret, o, h, l, c = _market(500, 0)
    tgt = _held(np.sign(np.random.default_rng(1).standard_normal(500)), 5)
    eng = s.BacktestOHLCTarget(taker_fee=2e-4)
    res = s.position_test(eng, tgt, o, h, l, c, n=20)
    eng.reset()
    summary = s.backtest_report(eng(tgt, o, h, l, c))[1]
    assert res.observed == pytest.approx(summary["sharpe"])
    res = s.position_test(eng, tgt, o, h, l, c, n=20, metric="total_pnl")
    assert res.observed == pytest.approx(summary["total_pnl"])
    res = s.position_test(eng, tgt, o, h, l, c, n=20, metric="max_drawdown")
    assert res.observed == pytest.approx(summary["max_drawdown"])


def test_p_value_bounds_and_result_fields():
    ret, o, h, l, c = _market(300, 2)
    tgt = _held(np.sign(np.random.default_rng(3).standard_normal(300)), 4)
    res = s.position_test(s.BacktestOHLCTarget(), tgt, o, h, l, c, n=99, seed=4)
    assert 1 / 100 <= res.p_value <= 1
    assert res.null.shape == (99,)
    assert res.metric == "sharpe" and res.method == "shift" and res.min_shift >= 1
    again = s.position_test(s.BacktestOHLCTarget(), tgt, o, h, l, c, n=99, seed=4)
    np.testing.assert_array_equal(res.null, again.null)


@pytest.mark.parametrize("method", ["shift", "segments"])
def test_detects_a_planted_edge(method):
    ret, o, h, l, c = _market(1500, 5)
    rng = np.random.default_rng(6)
    nxt = np.r_[np.sign(ret[1:]), 0.0]            # knows the next bar's direction
    tgt = np.where(rng.random(1500) < 0.6, nxt, -nxt)
    res = s.position_test(s.BacktestOHLCTarget(), tgt, o, h, l, c,
                          n=200, method=method, metric="total_pnl")
    assert res.p_value < 0.01


@pytest.mark.parametrize("method", ["shift", "segments"])
def test_calibrated_under_the_null(method):
    # A random signal on a random walk: p-values should be roughly uniform.
    ps = []
    for k in range(200):
        ret, o, h, l, c = _market(400, 100 + k)
        tgt = _held(np.sign(np.random.default_rng(1000 + k).standard_normal(400)), 5)
        ps.append(s.position_test(s.BacktestOHLCTarget(), tgt, o, h, l, c,
                                  n=49, method=method, seed=k).p_value)
    ps = np.array(ps)
    assert 0.40 < ps.mean() < 0.60
    assert np.mean(ps <= 0.10) < 0.20
    # Kolmogorov-Smirnov distance to the uniform distribution.
    q = np.sort(ps)
    d = np.max(np.abs(q - np.arange(1, len(q) + 1) / len(q)))
    assert d < 0.15


def test_callable_metric_and_direction():
    ret, o, h, l, c = _market(300, 7)
    tgt = _held(np.sign(np.random.default_rng(8).standard_normal(300)), 5)
    final_position = lambda out: out[-1, :, 2]
    res = s.position_test(s.BacktestOHLCTarget(), tgt, o, h, l, c, n=30,
                          metric=final_position)
    assert res.metric == "<lambda>"
    lo = s.position_test(s.BacktestOHLCTarget(), tgt, o, h, l, c, n=30,
                         metric="total_pnl", larger_is_better=False)
    hi = s.position_test(s.BacktestOHLCTarget(), tgt, o, h, l, c, n=30,
                         metric="total_pnl")
    np.testing.assert_array_equal(lo.null, hi.null)
    assert lo.p_value + hi.p_value >= 1.0        # the two tails cover everything


def test_works_with_other_target_engines():
    ret, o, h, l, c = _market(300, 9)
    tgt = _held(np.sign(np.random.default_rng(10).standard_normal(300)), 5)
    res = s.position_test(s.BacktestPriceTarget(spread=1e-4), tgt, c, n=20)
    assert np.isfinite(res.observed)
    size = np.ones(300)
    res = s.position_test(s.BacktestTradesTarget(), tgt, c, size, n=20)
    assert res.null.size == 20


def test_argument_checks():
    ret, o, h, l, c = _market(100, 11)
    tgt = np.sign(ret)
    eng = s.BacktestOHLCTarget()
    with pytest.raises(ValueError):
        s.position_test(eng, tgt, o, h, l, c, method="bootstrap")
    with pytest.raises(ValueError):
        s.position_test(eng, tgt, o, h, l, c, metric="nope")
    with pytest.raises(ValueError, match="larger_is_better"):
        s.position_test(eng, tgt, o, h, l, c, metric="turnover")
    with pytest.raises(ValueError):
        s.position_test(eng, tgt, o, h, l, c[:-1])
    with pytest.raises(ValueError):
        s.position_test(eng, tgt, o, h, l, c, n=0)
    with pytest.raises(ValueError):
        s.position_test(eng, tgt, o, h, l, c, seed=-1)
    with pytest.raises(ValueError):
        s.segment_shuffle(tgt, 2, seed=2**53)


# --- permute_bars / permute_returns ----------------------------------------------

O6 = np.array([100, 101, 99, 102, 103, 101.])
H6, L6 = O6 * 1.01, O6 * 0.98
C6 = np.array([100.5, 99.5, 101, 102.5, 101.5, 102])
X7 = np.array([np.nan, 10, 11, 10.5, 12, 11.5, 13.])


def test_permutation_golden_values():
    # The JS/WASM tests check the same values (with a 1e-12 relative tolerance,
    # since exp/log may differ in the last bit between math libraries).
    o, h, l, c = s.permute_bars(O6, H6, L6, C6, 2, seed=7)
    np.testing.assert_allclose(o[:, 1], [100.0, 101.00000000000003, 98.99999999999999,
        100.50246305418726, 101.99507389162568, 103.50738916256157], rtol=1e-12)
    np.testing.assert_allclose(c[:, 1], [100.5, 99.49999999999997, 101.00000000000003,
        100.99512218680589, 103.00492610837439, 101.99999999999996], rtol=1e-12)
    np.testing.assert_allclose(s.permute_returns(X7, 2, seed=7)[:, 0], [np.nan, 10.0,
        11.30434782608696, 10.833333333333336, 12.380952380952385, 11.81818181818182, 13.0],
        rtol=1e-12)


def _ohlc_walk(T, seed):
    rng = np.random.default_rng(seed)
    c = 100 * np.exp(np.cumsum(rng.standard_normal(T) * 0.01))
    o = np.r_[c[0], c[:-1]] * np.exp(rng.standard_normal(T) * 0.002)
    return o, np.maximum(o, c) * 1.003, np.minimum(o, c) * 0.997, c


def test_permute_bars_invariants():
    o, h, l, c = _ohlc_walk(200, 1)
    po, ph, pl, pc = s.permute_bars(o, h, l, c, 30, seed=3)
    assert po.shape == (200, 30)
    np.testing.assert_allclose(pc[-1], c[-1], rtol=1e-10)          # same last close
    np.testing.assert_allclose(po[0], o[0]); np.testing.assert_allclose(pc[0], c[0])
    assert (pl <= np.minimum(po, pc) * (1 + 1e-12)).all()
    assert (ph >= np.maximum(po, pc) * (1 - 1e-12)).all()
    gaps = np.sort(np.log(o[1:] / c[:-1]))
    shapes = np.sort(np.log(h[1:] / o[1:]))
    for j in range(30):
        np.testing.assert_allclose(np.sort(np.log(po[1:, j] / pc[:-1, j])), gaps, atol=1e-12)
        np.testing.assert_allclose(np.sort(np.log(ph[1:, j] / po[1:, j])), shapes, atol=1e-12)
    assert not np.allclose(pc[:, 0], c)                            # the path really changes


def test_permute_returns_invariants_and_differences():
    x = np.r_[np.nan, np.nan, 50 + np.cumsum(np.random.default_rng(2).standard_normal(100))]
    p = s.permute_returns(x, 10, seed=1)
    assert np.isnan(p[:2]).all()
    np.testing.assert_allclose(p[-1], x[-1], rtol=1e-10)
    for j in range(10):
        np.testing.assert_allclose(np.sort(np.diff(np.log(p[2:, j]))),
                                   np.sort(np.diff(np.log(x[2:]))), atol=1e-12)
    spread = np.random.default_rng(3).standard_normal(80).cumsum()   # crosses zero
    d = s.permute_returns(spread, 5, mode="diff", seed=2)
    np.testing.assert_allclose(np.sort(np.diff(d[:, 0])), np.sort(np.diff(spread)), atol=1e-12)
    with pytest.raises(ValueError):
        s.permute_returns(spread, 5)                                 # log needs positive values


def test_permutations_seeded_and_chunk_invariant():
    from screamer.screamer_bindings import _permute_bars, _permute_returns
    o, h, l, c = _ohlc_walk(60, 4)
    E = np.empty(0)
    full = _permute_bars(o, h, l, c, "log", 9, 0, 12, E, False)
    parts = [_permute_bars(o, h, l, c, "log", 9, a, b, E, False) for a, b in ((0, 5), (5, 7))]
    for k in range(5):
        np.testing.assert_array_equal(full[k], np.hstack([parts[0][k], parts[1][k]]))
    r = _permute_returns(c, "log", 9, 0, 8, E, False)
    r2 = [_permute_returns(c, "log", 9, a, b, E, False) for a, b in ((0, 3), (3, 5))]
    np.testing.assert_array_equal(r[0], np.hstack([r2[0][0], r2[1][0]]))


def test_permute_bars_rejects_bad_prices():
    o, h, l, c = _ohlc_walk(20, 5)
    with pytest.raises(ValueError):
        s.permute_bars(o, h, l, -c, 2)
    with pytest.raises(ValueError):
        s.permute_bars(o, h, l, c[:-1], 2)


# --- strategy_test ------------------------------------------------------------------

def _momentum_bars(phi, seed, T=1200):
    rng = np.random.default_rng(seed)
    e = rng.standard_normal(T) * 0.01
    r = np.zeros(T)
    for t in range(1, T):
        r[t] = phi * r[t - 1] + e[t]
    c = 100 * np.exp(np.cumsum(r))
    o = np.r_[c[0], c[:-1]] * np.exp(rng.standard_normal(T) * 0.001)
    return o, np.maximum(o, c) * 1.002, np.minimum(o, c) * 0.998, c


def _trend(o, h, l, c):
    sig = np.sign(np.nan_to_num(s.EwMean(span=10)(s.LogReturn()(c))))
    return s.BacktestOHLCTarget(taker_fee=1e-4)(sig, o, h, l, c)


def test_strategy_test_observed_and_methods():
    bars = _momentum_bars(0.15, 1)
    res = s.strategy_test(_trend, *bars, n=200)
    summary = s.backtest_report(_trend(*bars))[1]
    assert res.observed == pytest.approx(summary["sharpe"])
    assert res.p_value < 0.05 and res.method == "bars" and res.null.size == 200
    flat = s.strategy_test(_trend, *_momentum_bars(0.0, 1), n=200)
    assert flat.p_value > 0.05


def test_strategy_test_returns_method_and_pipeline():
    from screamer import Input, Pipeline, CombineLatest
    c = _momentum_bars(0.3, 2)[3]            # strong momentum: detectable for any seed
    p = Input("price")
    sig = s.Sign()(s.EwMean(span=10)(s.LogReturn()(p)))
    pipe = Pipeline(inputs=[p], outputs=[s.BacktestPriceTarget()(CombineLatest()(sig, p))])
    res = s.strategy_test(pipe, c, n=200, method="returns")
    assert res.p_value < 0.05


def test_strategy_test_shift_detects_informative_features():
    c = _momentum_bars(0.3, 3)[3]
    informative = np.nan_to_num(s.EwMean(span=10)(s.LogReturn()(c)))
    noise = np.random.default_rng(9).standard_normal(len(c))
    strat = lambda feat, price: s.BacktestPriceTarget()(np.sign(feat), price)
    assert s.strategy_test(strat, informative, c, n=200, method="shift",
                           columns=[0]).p_value < 0.05
    assert s.strategy_test(strat, noise, c, n=200, method="shift",
                           columns=[0]).p_value > 0.05


def test_other_inputs_move_with_their_bar_and_fixed_inputs_stay():
    c = _momentum_bars(0.0, 4, T=200)[3]
    lr = np.r_[np.nan, np.diff(np.log(c))]          # a per-bar column tied to price
    stamps = np.arange(200, dtype=float) * 60.0

    def check(price, step, ts):
        np.testing.assert_allclose(step[1:], np.diff(np.log(price)), atol=1e-12)
        np.testing.assert_array_equal(ts, stamps)
        return s.BacktestPriceTarget()(np.ones(len(price)), price)

    s.strategy_test(check, c, lr, stamps, n=5, method="returns", fixed=[2])


def test_strategy_test_calibrated_under_the_null():
    ps = []
    for k in range(150):
        bars = _momentum_bars(0.0, 500 + k, T=400)
        ps.append(s.strategy_test(_trend, *bars, n=49, seed=k).p_value)
    ps = np.array(ps)
    assert 0.40 < ps.mean() < 0.60
    assert np.mean(ps <= 0.10) < 0.20


def test_strategy_test_argument_checks():
    bars = _momentum_bars(0.0, 6, T=100)
    with pytest.raises(ValueError):
        s.strategy_test(_trend, *bars, method="bootstrap")
    with pytest.raises(ValueError):
        s.strategy_test(_trend, *bars, ohlc=(0, 1, 2))
    with pytest.raises(ValueError):
        s.strategy_test(_trend, *bars, method="shift")             # no columns
    with pytest.raises(ValueError):
        s.strategy_test(lambda *a: np.zeros((100, 3)), *bars, n=2)  # not an engine output
    with pytest.raises(ValueError):
        s.strategy_test(_trend, *bars, fixed=[7])


# --- position_test with several strategies (family-wise) ---------------------------

def test_family_single_column_matches_one_dimensional():
    ret, o, h, l, c = _market(400, 20)
    tgt = _held(np.sign(np.random.default_rng(21).standard_normal(400)), 5)
    one = s.position_test(s.BacktestOHLCTarget(), tgt, o, h, l, c, n=60, seed=2)
    fam = s.position_test(s.BacktestOHLCTarget(), tgt[:, None], o, h, l, c, n=60, seed=2)
    np.testing.assert_allclose(fam.null[:, 0], one.null)
    assert fam.p_value[0] == pytest.approx(one.p_value)
    assert fam.family_p_value[0] == pytest.approx(one.p_value)    # K = 1: no correction
    assert one.family_p_value is None


def test_family_p_value_corrects_for_picking_the_best():
    ret, o, h, l, c = _market(600, 22)
    rng = np.random.default_rng(23)
    tg = np.column_stack([_held(np.sign(rng.standard_normal(600)), 5) for _ in range(15)])
    res = s.position_test(s.BacktestOHLCTarget(), tg, o, h, l, c, n=199)
    assert res.null.shape == (199, 15) and res.observed.shape == (15,)
    assert np.all(res.family_p_value >= res.p_value - 1e-12)
    assert res.family_p_value[np.argmin(res.p_value)] > res.p_value.min()


def test_family_calibrated_under_the_null():
    # The best of K random strategies should look significant at the 10% level
    # in about 10% of trials once the family-wise p-value is used.
    hits = 0
    trials = 120
    for k in range(trials):
        ret, o, h, l, c = _market(300, 3000 + k)
        rng = np.random.default_rng(4000 + k)
        tg = np.column_stack([_held(np.sign(rng.standard_normal(300)), 5) for _ in range(8)])
        res = s.position_test(s.BacktestOHLCTarget(), tg, o, h, l, c, n=49, seed=k)
        hits += np.nanmin(res.family_p_value) <= 0.10
    assert hits / trials < 0.20


def test_family_requires_shift():
    ret, o, h, l, c = _market(100, 24)
    tg = np.sign(np.random.default_rng(25).standard_normal((100, 3)))
    with pytest.raises(ValueError):
        s.position_test(s.BacktestOHLCTarget(), tg, o, h, l, c, method="segments")


# --- probabilistic and deflated Sharpe ratios ---------------------------------------

def _psr_reference(x, bench):
    from scipy import stats
    sr = x.mean() / x.std(ddof=1)
    sk, ku = stats.skew(x), stats.kurtosis(x, fisher=False)
    return stats.norm.cdf((sr - bench) * np.sqrt(len(x) - 1)
                          / np.sqrt(1 - sk * sr + (ku - 1) / 4 * sr ** 2))


def test_probabilistic_sharpe_matches_formula():
    rng = np.random.default_rng(30)
    for k in range(5):
        x = rng.standard_t(4 + k, size=500 + 200 * k) * 0.01 + 0.0005 * k
        for bench in (0.0, 0.02):
            assert s.probabilistic_sharpe(x, bench) == pytest.approx(_psr_reference(x, bench),
                                                                     rel=1e-12, abs=1e-14)


def test_deflated_sharpe_matches_formula():
    from scipy import stats
    rng = np.random.default_rng(31)
    x = rng.standard_normal(800) * 0.01 + 0.001
    for N in (2, 10, 100, 1000):
        trials = rng.standard_normal(N) * 0.03
        g = 0.5772156649015329
        sr0 = trials.std(ddof=1) * ((1 - g) * stats.norm.ppf(1 - 1 / N)
                                    + g * stats.norm.ppf(1 - 1 / (N * np.e)))
        p, bench = s.deflated_sharpe(x, trials)
        assert bench == pytest.approx(sr0, rel=1e-12)
        assert p == pytest.approx(_psr_reference(x, sr0), rel=1e-12, abs=1e-14)


def test_sharpe_tests_accept_engine_output_and_skip_nan():
    rng = np.random.default_rng(32)
    pnl = rng.standard_normal(300) * 0.01
    out = np.column_stack([np.cumsum(pnl), pnl, np.ones(300), np.zeros(300)])
    assert s.probabilistic_sharpe(out) == pytest.approx(s.probabilistic_sharpe(pnl))
    holed = np.r_[np.nan, pnl, np.nan]
    assert s.probabilistic_sharpe(holed) == pytest.approx(s.probabilistic_sharpe(pnl))


def test_more_trials_deflate_more():
    rng = np.random.default_rng(33)
    x = rng.standard_normal(1000) * 0.01 + 0.0008
    few = s.deflated_sharpe(x, rng.standard_normal(3) * 0.03)[0]
    many = s.deflated_sharpe(x, rng.standard_normal(3000) * 0.03)[0]
    assert many < few < s.probabilistic_sharpe(x)


def test_sharpe_argument_checks():
    with pytest.raises(ValueError):
        s.probabilistic_sharpe([0.1, 0.2])                       # too short
    with pytest.raises(ValueError):
        s.probabilistic_sharpe(np.zeros(10))                     # no dispersion
    with pytest.raises(ValueError):
        s.deflated_sharpe(np.random.default_rng(0).standard_normal(50), [0.1])


# --- permutation modes and volume handling ----------------------------------------

def test_diff_mode_bars_allow_negative_prices():
    o, h, l, c = (x - 100.0 for x in _ohlc_walk(150, 40))       # crosses zero
    po, ph, pl, pc = s.permute_bars(o, h, l, c, 20, mode="diff", seed=1)
    np.testing.assert_allclose(pc[-1], c[-1], atol=1e-9)
    assert (pl <= np.minimum(po, pc) + 1e-9).all() and (ph >= np.maximum(po, pc) - 1e-9).all()
    for j in range(20):
        np.testing.assert_allclose(np.sort(po[1:, j] - pc[:-1, j]), np.sort(o[1:] - c[:-1]),
                                   atol=1e-9)
    with pytest.raises(ValueError):
        s.permute_bars(o, h, l, c, 2)                              # log needs positive prices


def test_level_mode_shuffles_whole_rows():
    o, h, l, c = _ohlc_walk(80, 41)
    po, ph, pl, pc = s.permute_bars(o, h, l, c, 10, mode="level", seed=2)
    rows = {tuple(r) for r in np.column_stack([o, h, l, c])}
    for j in range(10):
        got = np.column_stack([po[:, j], ph[:, j], pl[:, j], pc[:, j]])
        assert {tuple(r) for r in got} == rows
    x = np.r_[np.nan, np.random.default_rng(3).standard_normal(50)]
    px = s.permute_returns(x, 5, mode="level", seed=3)
    assert np.isnan(px[0]).all()
    for j in range(5):
        np.testing.assert_array_equal(np.sort(px[1:, j]), np.sort(x[1:]))


def test_volume_keep_and_notional():
    from screamer.screamer_bindings import _permute_bars, _permute_returns
    o, h, l, c = _ohlc_walk(100, 42)
    vol = np.random.default_rng(4).uniform(1, 9, 100)
    *_, kv = s.permute_bars(o, h, l, c, 6, volume=vol, seed=5)
    *px, nv = s.permute_bars(o, h, l, c, 6, volume=vol, volume_mode="notional", seed=5)
    src = _permute_bars(o, h, l, c, "log", 5, 0, 6, vol, True)[4].astype(int)
    np.testing.assert_allclose(kv, vol[src])                      # keep: the source volume
    typ, ptyp = (h + l + c) / 3, (px[1] + px[2] + px[3]) / 3
    np.testing.assert_allclose(nv * ptyp, (vol * typ)[src])       # notional: same amount
    p, pv = s.permute_returns(c, 6, volume=vol, volume_mode="notional", seed=5)
    src = _permute_returns(c, "log", 5, 0, 6, vol, True)[1].astype(int)
    np.testing.assert_allclose(pv * p, (vol * c)[src])
    lv = s.permute_bars(o, h, l, c, 3, mode="level", volume=vol, volume_mode="notional")
    src = _permute_bars(o, h, l, c, "level", 0, 0, 3, vol, True)[4].astype(int)
    np.testing.assert_allclose(lv[4], vol[src])                   # whole bars: unchanged


def test_mode_argument_checks():
    o, h, l, c = _ohlc_walk(20, 43)
    with pytest.raises(ValueError):
        s.permute_bars(o, h, l, c, 2, mode="pct")
    with pytest.raises(ValueError):
        s.permute_bars(o, h, l, c, 2, volume=np.ones(20), volume_mode="dollars")
    with pytest.raises(ValueError):
        s.permute_bars(o, h, l, c, 2, volume=np.ones(19))
    with pytest.raises(ValueError):                               # notional needs prices > 0
        s.permute_returns(c - 200, 2, mode="diff", volume=np.ones(20), volume_mode="notional")


def test_strategy_test_moves_volume_with_its_mode():
    o, h, l, c = _ohlc_walk(200, 44)
    vol = np.random.default_rng(6).uniform(1, 9, 200)
    amount = vol * (h + l + c) / 3

    def check(o_, h_, l_, c_, v_):
        np.testing.assert_allclose(np.sort(v_ * (h_ + l_ + c_) / 3), np.sort(amount))
        return s.BacktestOHLCTarget()(np.ones(len(c_)), o_, h_, l_, c_)

    s.strategy_test(check, o, h, l, c, vol, n=4, volume=4, volume_mode="notional")
    lv = s.strategy_test(check, o, h, l, c, vol, n=4, mode="level", volume=4)
    assert lv.null.size == 4
    with pytest.raises(ValueError):
        s.strategy_test(check, o, h, l, c, vol, method="shift", columns=[4], volume=4)


def test_mode_golden_values():
    # Checked by the JS/WASM tests too.
    V6 = np.array([5, 3, 8, 2, 6, 4.])
    d = s.permute_bars(O6 - 100, H6 - 100, L6 - 100, C6 - 100, 2, mode="diff", seed=7)
    np.testing.assert_allclose(d[3][:, 1], [0.5, -0.5, 1.0, 1.0, 3.0, 2.0], atol=1e-12)
    lv = s.permute_bars(O6, H6, L6, C6, 2, mode="level", seed=7)
    np.testing.assert_array_equal(lv[3][:, 0], [101.5, 102.5, 101.0, 99.5, 100.5, 102.0])
    nv = s.permute_bars(O6, H6, L6, C6, 2, seed=7, volume=V6, volume_mode="notional")
    np.testing.assert_allclose(nv[4][:, 1], [5.0, 3.0, 8.0, 2.029800999901968,
                                             3.9609756097560966, 5.97058823529412], rtol=1e-12)


# --- per-run statistics ----------------------------------------------------

def test_stats_match_backtest_report_and_align_with_null():
    ret, o, h, l, c = _market(600, 3)
    tgt = np.sign(np.nan_to_num(s.EwMean(span=8)(ret)))
    eng = s.BacktestOHLCTarget(taker_fee=1e-4)
    res = s.position_test(eng, tgt, o, h, l, c, n=40, seed=5, periods_per_year=252)
    _, summary = s.backtest_report(eng(tgt, o, h, l, c), periods_per_year=252)
    assert set(res.observed_stats) == set(summary) == set(res.null_stats)
    for k, v in summary.items():
        np.testing.assert_allclose(res.observed_stats[k], v, equal_nan=True)
    assert res.observed == res.observed_stats["sharpe"]
    for v in res.null_stats.values():
        assert v.shape == (40,)
    sh = res.null_stats["sharpe"]
    np.testing.assert_array_equal(res.null, sh[np.isfinite(sh)])
    # Each null run's statistics are those of backtest_report on that run.
    surr = s.circular_shift(tgt, 40, min_shift=res.min_shift, seed=5)
    for j in (0, 17, 39):
        eng.reset()
        _, sj = s.backtest_report(eng(surr[:, j], o, h, l, c), periods_per_year=252)
        for k, v in sj.items():
            np.testing.assert_allclose(res.null_stats[k][j], v, equal_nan=True)


def test_any_summary_statistic_is_a_metric():
    ret, o, h, l, c = _market(600, 3)
    tgt = np.sign(np.nan_to_num(s.EwMean(span=8)(ret)))
    eng = s.BacktestOHLCTarget(taker_fee=1e-4)
    res = s.position_test(eng, tgt, o, h, l, c, n=30, metric="win_rate")
    assert res.metric == "win_rate" and res.observed == res.observed_stats["win_rate"]
    wr = res.null_stats["win_rate"]
    np.testing.assert_array_equal(res.null, wr[np.isfinite(wr)])
    low = s.position_test(eng, tgt, o, h, l, c, n=30, metric="turnover",
                          larger_is_better=False)
    hits = np.count_nonzero(low.null <= low.observed)
    assert low.p_value == (1 + hits) / (1 + low.null.size)


def test_family_and_strategy_stats_shapes():
    ret, o, h, l, c = _market(500, 4)
    tgts = np.column_stack([np.sign(np.nan_to_num(s.EwMean(span=k)(ret))) for k in (4, 16)])
    eng = s.BacktestOHLCTarget()
    fam = s.position_test(eng, tgts, o, h, l, c, n=20)
    assert fam.observed_stats["sortino"].shape == (2,)
    assert fam.null_stats["sortino"].shape == (20, 2)
    np.testing.assert_array_equal(fam.null, fam.null_stats["sharpe"])

    def strategy(o, h, l, c):
        sig = np.sign(np.nan_to_num(s.EwMean(span=8)(s.LogReturn()(c))))
        return s.BacktestOHLCTarget()(sig, o, h, l, c)

    st = s.strategy_test(strategy, o, h, l, c, n=15, metric="calmar", seed=2)
    _, summary = s.backtest_report(strategy(o, h, l, c))
    for k, v in summary.items():
        np.testing.assert_allclose(st.observed_stats[k], v, equal_nan=True)
    assert st.null_stats["calmar"].shape == (15,)
    cm = st.null_stats["calmar"]
    np.testing.assert_array_equal(st.null, cm[np.isfinite(cm)])


# --- strategy_test on (T, K) portfolio inputs (tracker C2) ------------------

from screamer.screamer_bindings import _permute_bars, _permute_returns  # noqa: E402

_NO_VOL = np.empty(0)


def _panel(T=300, K=4, seed=0):
    rng = np.random.default_rng(seed)
    c = 100 * np.exp(np.cumsum(0.01 * rng.standard_normal((T, K)), axis=0))
    o = np.vstack([c[:1], c[:-1]]) * np.exp(0.001 * rng.standard_normal((T, K)))
    h = np.maximum(o, c) * 1.002
    l = np.minimum(o, c) * 0.998
    return o, h, l, c


def test_strategy_test_accepts_portfolio_inputs():
    rng = np.random.default_rng(0); T, K = 300, 4
    c = 100 * np.exp(np.cumsum(0.01 * rng.standard_normal((T, K)), axis=0))
    def book(c):
        w = np.full((T, K), 1.0 / K)
        return s.BacktestPortfolioTarget()(w, c)
    res = s.strategy_test(book, c, method="returns", n=20)
    assert np.isfinite(res.p_value)


@pytest.mark.parametrize("mode", ["log", "diff", "level"])
def test_strategy_test_portfolio_bars_and_returns_every_mode(mode):
    o, h, l, c = _panel()
    T, K = c.shape
    seen = []

    def book_bars(o, h, l, c):
        assert o.shape == (T, K)
        seen.append(c)
        w = np.sign(np.nan_to_num(s.EwMean(span=5)(s.LogReturn()(c)))) / K
        return s.BacktestPortfolioTarget()(w, c)

    res = s.strategy_test(book_bars, o, h, l, c, n=8, mode=mode, seed=1)
    assert res.null.size == 8 and np.isfinite(res.p_value)
    # every copy is a (T, K) panel that differs from the real one
    assert all(x.shape == (T, K) for x in seen)
    assert not np.array_equal(seen[1], c)

    res = s.strategy_test(lambda c: book_bars(c, c, c, c), c, method="returns",
                          n=8, mode=mode, seed=1)
    assert res.null.size == 8 and np.isfinite(res.p_value)


@pytest.mark.parametrize("mode", ["log", "diff", "level"])
def test_shared_permutation_preserves_cross_correlation(mode):
    # A (T, 2) block with identical columns: every permuted copy keeps the two
    # columns identical, so one permutation is applied to all assets.
    _, _, _, c = _panel(T=120, K=1, seed=3)
    x = np.ascontiguousarray(np.hstack([c, c]))
    out, src = _permute_returns(x, mode, 5, 0, 6, _NO_VOL, False)
    assert out.shape == (120, 6, 2) and src.shape == (120, 6)
    np.testing.assert_array_equal(out[:, :, 0], out[:, :, 1])

    o, h, l, c = (np.ascontiguousarray(np.hstack([a, a])) for a in _panel(T=120, K=1))
    res = _permute_bars(o, h, l, c, mode, 5, 0, 6, _NO_VOL, False)
    for a in res[:4]:
        assert a.shape == (120, 6, 2)
        np.testing.assert_array_equal(a[:, :, 0], a[:, :, 1])
    assert res[4].shape == (120, 6)


@pytest.mark.parametrize("mode", ["log", "diff", "level"])
def test_panel_kernel_column_equals_series_kernel(mode):
    # Without a NaN warmup, asset k of a (T, K) block gets exactly the copies
    # that the 1-D kernel gives for that asset alone: the same permutation.
    o, h, l, c = _panel(T=90, K=3, seed=4)
    v = np.ascontiguousarray(1000 + np.arange(90 * 3, dtype=float).reshape(90, 3))
    out = _permute_returns(c, mode, 9, 2, 5, v, True)
    bars = _permute_bars(o, h, l, c, mode, 9, 2, 5, v, True)
    for k in range(3):
        one = _permute_returns(np.ascontiguousarray(c[:, k]), mode, 9, 2, 5,
                               np.ascontiguousarray(v[:, k]), True)
        np.testing.assert_array_equal(out[0][:, :, k], one[0])
        np.testing.assert_array_equal(out[1], one[1])
        np.testing.assert_array_equal(out[2][:, :, k], one[2])
        col = [np.ascontiguousarray(a[:, k]) for a in (o, h, l, c, v)]
        one = _permute_bars(*col[:4], mode, 9, 2, 5, col[4], True)
        for q in (0, 1, 2, 3, 5):
            np.testing.assert_array_equal(bars[q][:, :, k], one[q])
        np.testing.assert_array_equal(bars[4], one[4])


def test_panel_returns_warmup_is_the_longest_prefix():
    _, _, _, c = _panel(T=60, K=2, seed=6)
    c = c.copy()
    c[:3, 0] = np.nan
    c[:7, 1] = np.nan
    out, src = _permute_returns(c, "log", 1, 0, 4, _NO_VOL, False)
    # rows before every asset has a value stay in place
    np.testing.assert_array_equal(out[:7], np.repeat(c[:7, None, :], 4, axis=1))
    np.testing.assert_array_equal(src[:7], np.repeat(np.arange(7.0)[:, None], 4, axis=1))
    # each asset keeps its first valid value and last value
    np.testing.assert_allclose(out[-1], np.repeat(c[-1:], 4, axis=0), rtol=1e-12)
    assert np.all(np.isfinite(out[7:]))


def test_strategy_test_mixed_inputs_move_with_the_bar():
    o, h, l, c = _panel(T=150, K=3, seed=8)
    feature = np.arange(150, dtype=float)         # (T,) per-bar data
    got = []

    def book(c, f):
        got.append((c, f))
        return s.BacktestPortfolioTarget()(np.full(c.shape, 1 / 3), c)

    s.strategy_test(book, c, feature, method="returns", n=3, mode="level", seed=2)
    out, src = _permute_returns(c, "level", 2, 0, 3, _NO_VOL, False)
    for j, (cj, fj) in enumerate(got[1:]):
        np.testing.assert_array_equal(cj, out[:, j])
        np.testing.assert_array_equal(fj, feature[src[:, j].astype(int)])

    # "shift" rotates a (T, K) feature with one offset for all columns
    feats = np.column_stack([np.arange(150.0), 1000 + np.arange(150.0)])
    got.clear()
    s.strategy_test(lambda c, f: book(c, f), c, feats, method="shift", columns=[1],
                    n=3, seed=2)
    for cj, fj in got[1:]:
        np.testing.assert_array_equal(cj, c)
        np.testing.assert_array_equal(fj[:, 1] - fj[:, 0], 1000.0)


def test_strategy_test_portfolio_volume():
    o, h, l, c = _panel(T=100, K=2, seed=9)
    vol = np.full((100, 2), 50.0)
    seen = []

    def book(o, h, l, c, v):
        seen.append(v)
        return s.BacktestPortfolioTarget()(np.full(c.shape, 0.5), c)

    s.strategy_test(book, o, h, l, c, vol, n=3, volume=4, volume_mode="notional")
    assert seen[1].shape == (100, 2) and not np.array_equal(seen[1], vol)
    # a (T,) volume with (T, K) prices moves with its bar in "keep" mode
    seen.clear()
    s.strategy_test(book, o, h, l, c, vol[:, 0], n=3, volume=4)
    np.testing.assert_array_equal(seen[1], vol[:, 0])
    with pytest.raises(ValueError, match="notional"):
        s.strategy_test(book, o, h, l, c, vol[:, 0], n=3, volume=4, volume_mode="notional")


def test_strategy_test_portfolio_argument_checks():
    o, h, l, c = _panel(T=50, K=2)
    book = lambda *x: s.BacktestPortfolioTarget()(np.full((50, 2), 0.5), x[-1])  # noqa: E731
    with pytest.raises(ValueError, match="same length"):
        s.strategy_test(book, c, c[:40], method="returns", n=2)
    with pytest.raises(ValueError, match="1-D \\(T,\\) or 2-D"):
        s.strategy_test(book, c[:, :, None], method="returns", n=2)
    with pytest.raises(ValueError, match="1-D \\(T,\\) or 2-D"):
        s.strategy_test(book, np.float64(1.0), method="returns", n=2)
    with pytest.raises(TypeError):
        s.strategy_test(book, "abc", method="returns", n=2)
    with pytest.raises(TypeError):
        s.strategy_test(book, None, method="returns", n=2)
    with pytest.raises(ValueError, match="same shape"):
        s.strategy_test(book, o, h, l[:, 0], c, n=2)


# strategy_test with 1-D inputs, n=70 (past the 64-copy chunk), seed=3,
# computed with screamer 2.8.0: (observed, p_value, null size, null sum,
# null at positions 0, 1, 63, 64, 65, -1).
_V280 = {
    "returns_log": (0.062141750284824174, 0.2535211267605634, 70, 1.3263899545305522,
        [-0.12758338148160225, 0.03140848583704488, -0.05009701958086175, -0.015659826465689305, -0.010975036415169072, -0.07237309884376006]),
    "returns_level": (0.062141750284824174, 0.014084507042253521, 70, -41.96035123118885,
        [-0.5581525153014752, -0.6119503188604324, -0.5199677261769974, -0.6935172724528123, -0.6122945156166879, -0.592000830427454]),
    "bars_diff_notional": (0.07551555188387185, 0.2112676056338028, 70, 1.0176196426613808,
        [-0.04832591955453531, 0.03559172027890768, -0.12526888355379764, -0.10223626983630979, 0.021836781851415715, -0.039161682929031986]),
    "shift": (0.07551555188387185, 0.29577464788732394, 70, 3.220558304861609,
        [0.12292252240556496, 0.08372682387027214, 0.11017250678640951, 0.05841980382586971, 0.08376068436457057, 0.00432638354768181]),
}


def test_1d_results_unchanged():
    rng = np.random.default_rng(7); T = 200
    c = 100 * np.exp(np.cumsum(0.01 * rng.standard_normal(T)))
    o = np.r_[c[0], c[:-1]] * np.exp(0.001 * rng.standard_normal(T))
    h = np.maximum(o, c) * 1.002; l = np.minimum(o, c) * 0.998
    v = 1000 + 100 * rng.random(T)

    def strat_r(c):
        sig = np.sign(np.nan_to_num(s.EwMean(span=5)(s.LogReturn()(c))))
        return s.BacktestPriceTarget()(sig, c)

    def strat_b(o, h, l, c, v):
        sig = np.sign(np.nan_to_num(s.EwMean(span=5)(s.LogReturn()(c)))) * (v > 1050)
        return s.BacktestOHLCTarget()(sig, o, h, l, c)

    cases = {
        "returns_log": (strat_r, (c,), dict(method="returns")),
        "returns_level": (strat_r, (c,), dict(method="returns", mode="level")),
        "bars_diff_notional": (strat_b, (o, h, l, c, v),
                               dict(mode="diff", volume=4, volume_mode="notional")),
        "shift": (strat_b, (o, h, l, c, v), dict(method="shift", columns=[4])),
    }
    for name, (f, x, kw) in cases.items():
        r = s.strategy_test(f, *x, n=70, seed=3, **kw)
        observed, p_value, size, total, picks = _V280[name]
        # rtol, not equality: log mode goes through the platform's exp and log
        np.testing.assert_allclose(r.observed, observed, rtol=1e-12, err_msg=name)
        assert r.p_value == p_value and r.null.size == size, name
        np.testing.assert_allclose(r.null.sum(), total, rtol=1e-11, err_msg=name)
        np.testing.assert_allclose(r.null[[0, 1, 63, 64, 65, -1]], picks, rtol=1e-12,
                                   err_msg=name)


def test_panel_kernels_read_non_contiguous_input():
    o, h, l, c = _panel(T=80, K=6, seed=11)
    views = {
        "fortran": np.asfortranarray(c),
        "strided": c[:, ::2],
        "reversed": c[::-1, ::-1][::-1, ::-1],
    }
    for name, v in views.items():
        got = _permute_returns(v, "log", 4, 0, 5, _NO_VOL, False)
        ref = _permute_returns(np.ascontiguousarray(v), "log", 4, 0, 5, _NO_VOL, False)
        for g, r in zip(got, ref):
            np.testing.assert_array_equal(g, r, err_msg=name)
    f = [np.asfortranarray(a) for a in (o, h, l, c)]
    got = _permute_bars(*f, "diff", 4, 0, 5, _NO_VOL, False)
    ref = _permute_bars(o, h, l, c, "diff", 4, 0, 5, _NO_VOL, False)
    for g, r in zip(got, ref):
        np.testing.assert_array_equal(g, r)
    # and through strategy_test
    seen = []

    def book(c):
        seen.append(c)
        return s.BacktestPortfolioTarget()(np.full(c.shape, 0.5), c)

    s.strategy_test(book, views["strided"], method="returns", n=3, seed=1)
    a = seen[1:]
    seen.clear()
    s.strategy_test(book, np.ascontiguousarray(views["strided"]), method="returns", n=3, seed=1)
    for x, y in zip(a, seen[1:]):
        np.testing.assert_array_equal(x, y)


def test_public_permute_functions_reject_panels():
    o, h, l, c = _panel(T=40, K=2)
    with pytest.raises(ValueError, match="x must be a 1-D series"):
        s.permute_returns(c, 3)
    with pytest.raises(ValueError, match="open must be a 1-D series"):
        s.permute_bars(o, h, l, c, 3)
    with pytest.raises(ValueError, match="volume must be a 1-D series"):
        s.permute_returns(c[:, 0], 3, volume=c)


def test_strategy_test_zero_assets():
    empty = np.empty((30, 0))
    with pytest.raises(ValueError, match="at least one asset"):
        s.strategy_test(lambda c: np.zeros((30, 4)), empty, method="returns", n=2)


# --- strategy_test driver in shared C++ (tracker B5) -------------------------

class _StrategyFailed(Exception):
    pass


def test_strategy_test_callback_errors_propagate_unchanged():
    c = _panel(T=60, K=1, seed=12)[3][:, 0]
    raised = _StrategyFailed("from the strategy")
    calls = []

    def fails_late(price):
        calls.append(price)
        if len(calls) == 3:
            raise raised
        return s.BacktestPriceTarget()(np.ones(len(price)), price)

    with pytest.raises(_StrategyFailed) as info:
        s.strategy_test(fails_late, c, method="returns", n=5)
    assert info.value is raised and len(calls) == 3
    with pytest.raises(ZeroDivisionError):
        s.strategy_test(lambda p: s.BacktestPriceTarget()(np.ones(60), p), c,
                        method="returns", n=2, metric=lambda out: 1 / 0)
    # the inputs are not held on to after a failed run
    before = sys.getrefcount(c)
    for _ in range(50):
        with pytest.raises(_StrategyFailed):
            s.strategy_test(lambda p: (_ for _ in ()).throw(_StrategyFailed()), c,
                            method="returns", n=2)
    assert sys.getrefcount(c) == before


def test_strategy_test_moved_inputs_are_new_arrays():
    o, h, l, c = _panel(T=50, K=2, seed=13)
    f = np.arange(50.0)
    seen = []

    def book(price, feature):
        seen.append((price, feature))
        return s.BacktestPortfolioTarget()(np.full(price.shape, 0.5), price)

    s.strategy_test(book, c, f, method="returns", n=3, seed=1)
    assert seen[0][0] is c and seen[0][1] is f          # the real run gets the inputs
    for (p1, f1), (p2, f2) in zip(seen[1:], seen[2:]):
        assert not np.shares_memory(p1, p2) and not np.shares_memory(f1, f2)


def _book_stats(out, periods_per_year):
    """Every statistic of one portfolio run, from NumPy: the book summed over
    the assets, turnover and trades from PortfolioReport (the 2.8.0 Python
    scorer, kept here as a reference)."""
    from screamer.screamer_bindings import _strategy_stats

    def last_finite(a):
        finite = np.isfinite(a)
        last = a.shape[0] - 1 - np.argmax(finite[::-1], axis=0)
        v = a[last, np.arange(a.shape[1])]
        v[~finite.any(axis=0)] = np.nan
        return v

    out = np.ascontiguousarray(out, dtype=float)
    book = [out[:, :, 0].sum(axis=1), out[:, :, 1].sum(axis=1),
            np.abs(out[:, :, 2]).sum(axis=1), out[:, :, 3].sum(axis=1)]
    report = np.asarray(s.BacktestReport(periods_per_year)(*book), dtype=float)
    report[:, 2:4] = np.asarray(s.PortfolioReport()(out), dtype=float)[:, 2:4]
    src = {"engine": np.column_stack(book)[:, None, :], "report": report[:, None, :],
           "stats": np.asarray(s.BacktestStats(periods_per_year)(*book), dtype=float)[:, None, :]}
    return {name: last_finite(src[q][:, :, c]) for name, q, c, _ in _strategy_stats()}


@pytest.mark.parametrize("K", [2, 9, 130])
def test_strategy_test_book_sum_matches_numpy(K):
    # A portfolio is scored on the book summed over assets in NumPy's
    # pairwise order, so the statistics equal those of the NumPy book.
    c = _panel(T=80, K=K, seed=K)[3]
    outs = []

    def book(price):
        w = np.sign(np.sin(np.arange(80.0)))[:, None] * np.full(price.shape, 1.0 / K)
        out = s.BacktestPortfolioTarget(fee=1e-4)(w, price)
        outs.append(out)
        return out

    res = s.strategy_test(book, c, method="returns", n=4, seed=2)
    for j, out in enumerate(outs):
        ref = {k: float(v[0]) for k, v in _book_stats(out, 1.0).items()}
        got = res.observed_stats if j == 0 else {k: v[j - 1] for k, v in res.null_stats.items()}
        for k in ref:
            assert np.array_equal(got[k], ref[k], equal_nan=True), (K, j, k)


@pytest.mark.parametrize("mode", ["log", "diff", "level"])
@pytest.mark.parametrize("K", [1, 3])
def test_kernel_copies_do_not_depend_on_the_tile(mode, K):
    # The kernels write tiles of 8 copies per pass; each copy must equal the
    # same copy generated alone (13 copies from column 5: one full tile and a
    # partial one).
    o, h, l, c = _panel(T=90, K=K, seed=14)
    v = np.ascontiguousarray(1000 + np.arange(90.0 * K).reshape(90, K))
    x = c.copy()
    x[:4, 0] = np.nan
    if K == 1:
        o, h, l, c, v, x = (np.ascontiguousarray(a[:, 0]) for a in (o, h, l, c, v, x))
    bars = _permute_bars(o, h, l, c, mode, 9, 5, 13, v, True)
    rets = _permute_returns(x, mode, 9, 5, 13, v, True)
    for j in range(13):
        one = _permute_bars(o, h, l, c, mode, 9, 5 + j, 1, v, True)
        for got, ref in zip(bars, one):
            np.testing.assert_array_equal(got[:, j], ref[:, 0])
        one = _permute_returns(x, mode, 9, 5 + j, 1, v, True)
        for got, ref in zip(rets, one):
            np.testing.assert_array_equal(got[:, j], ref[:, 0])


# --- B5 fix round 1: argument ranges and the statistic table ------------------

def test_strategy_test_range_of_n_and_integer_arguments():
    c = 100 + np.arange(40.0)
    hold = lambda p: s.BacktestPriceTarget()(np.ones(40), p)  # noqa: E731
    with pytest.raises(ValueError, match="^n must be at least 1$"):
        s.strategy_test(hold, c, method="returns", n=0)
    for n in (2**32 + 3, 2**60, 2**63, 2**80):
        with pytest.raises(ValueError, match=r"^n is too large: at most 2147483647 runs$"):
            s.strategy_test(hold, c, method="returns", n=n)
    with pytest.raises(TypeError, match="n must be an integer, not bool"):
        s.strategy_test(hold, c, method="returns", n=True)
    assert s.strategy_test(hold, c, method="returns", n=np.int64(3)).null.size == 3
    # integers outside int64 fail their own range check, not the signature
    with pytest.raises(ValueError, match="price refers to an input"):
        s.strategy_test(hold, c, method="returns", price=2**70)
    with pytest.raises(ValueError, match="fixed refers to an input"):
        s.strategy_test(hold, c, method="returns", fixed=[2**70])
    with pytest.raises(ValueError, match="volume refers to an input"):
        s.strategy_test(hold, c, method="returns", volume=-2**70)
    with pytest.raises(ValueError, match="min_shift must be in"):
        s.strategy_test(lambda f, p: hold(p), c, c, method="shift", columns=[0],
                        min_shift=-2**70)


def test_statistic_table_comes_from_cpp():
    from screamer.screamer_bindings import _strategy_stats
    table = {name: (q, c, larger) for name, q, c, larger in _strategy_stats()}
    assert table["sharpe"] == ("report", 5, True)
    assert table["turnover"] == ("report", 2, None)
    assert table["total_cost"] == ("report", 1, False)
    c = 100 + np.arange(40.0)
    res = s.strategy_test(lambda p: s.BacktestPriceTarget()(np.ones(40), p), c,
                          method="returns", n=3)
    assert list(res.observed_stats) == list(res.null_stats) == list(table)
    ret, o, h, l, c = _market(100, 1)
    res = s.position_test(s.BacktestPriceTarget(), np.sign(ret), c, n=3)
    assert list(res.observed_stats) == list(res.null_stats) == list(table)


# --- B6: position_test and walk_forward in the shared C++ driver ---------------

class _Engine:
    """A target engine that records what it gets and can fail on a given call."""

    def __init__(self, fail_on=None, error=None, out=None):
        self.engine = s.BacktestPriceTarget(fee=1e-4)
        self.calls, self.fail_on, self.error, self.out = [], fail_on, error, out

    def reset(self):
        self.engine.reset()

    def __call__(self, target, *market):
        self.calls.append(target)
        if self.fail_on is not None and len(self.calls) == self.fail_on:
            raise self.error
        return self.engine(target, *market) if self.out is None else self.out


def test_position_test_callback_errors_keep_their_identity():
    ret, o, h, l, c = _market(200, 31)
    tgt = _held(np.sign(ret), 4)
    for fail_on in (1, 2):                        # the real run, then the null
        err = KeyError("engine")
        with pytest.raises(KeyError) as info:
            s.position_test(_Engine(fail_on, err), tgt, c, n=5)
        assert info.value is err
        err = KeyError("engine")                  # the whole history, then the stitched run
        with pytest.raises(KeyError) as info:
            s.walk_forward(_Engine(fail_on, err), tgt, c, train=50, test=10)
        assert info.value is err
    seen = []
    err = RuntimeError("metric")

    def metric(out):
        seen.append(out.shape)
        if len(seen) == 2:                        # on the null, after the real run
            raise err
        return out[-1, :, 0]

    for target in (np.sign(np.sin(np.arange(200) / 7)), np.column_stack([tgt, -tgt])):
        seen.clear()
        with pytest.raises(RuntimeError) as info:
            s.position_test(s.BacktestPriceTarget(), target, c, n=3, metric=metric)
        assert info.value is err
    with pytest.raises(RuntimeError) as info:
        s.walk_forward(s.BacktestPriceTarget(), np.column_stack([tgt, -tgt]), c, train=50, test=10,
                       metric=lambda out: (_ for _ in ()).throw(err))
    assert info.value is err


def test_position_test_refcounts_are_stable_over_failures():
    ret, o, h, l, c = _market(120, 32)
    tgt = _held(np.sign(ret), 4)
    eng = _Engine(2, ValueError("boom"))
    before = (sys.getrefcount(tgt), sys.getrefcount(c), sys.getrefcount(eng))
    for _ in range(50):
        eng.calls.clear()
        eng.error = ValueError("boom")            # a new error: a raised one keeps its frames
        with pytest.raises(ValueError, match="boom"):
            s.position_test(eng, tgt, c, n=4)
        eng.calls.clear()
        eng.error = ValueError("boom")
        with pytest.raises(ValueError, match="boom"):
            s.walk_forward(eng, tgt, c, train=50, test=10)
    eng.calls.clear()
    eng.error = None
    assert (sys.getrefcount(tgt), sys.getrefcount(c), sys.getrefcount(eng)) == before


def test_position_test_engine_gets_new_target_arrays():
    ret, o, h, l, c = _market(120, 33)
    tgt = _held(np.sign(ret), 4)
    eng = _Engine()
    s.position_test(eng, tgt, c, n=6)
    assert eng.calls[0] is tgt and eng.calls[1].shape == (120, 6)
    eng = _Engine()
    W = np.column_stack([tgt, -tgt]) / 2
    port = s.BacktestPortfolioTarget()
    seen = []
    s.position_test(port, W, np.column_stack([c, c]), n=3,
                    metric=lambda out: (seen.append(out), np.nansum(out[:, :, 1]))[1])
    assert len({id(x) for x in seen}) == 4
    res = s.walk_forward(eng, np.column_stack([tgt, -tgt]), c, train=50, test=10)
    assert res.target is eng.calls[-1] and not np.shares_memory(res.target, eng.calls[0])


def test_position_test_range_of_n():
    ret, o, h, l, c = _market(60, 34)
    tgt = np.sign(ret)
    for target in (tgt, np.column_stack([tgt, -tgt])):
        with pytest.raises(ValueError, match="^n must be at least 1$"):
            s.position_test(s.BacktestPriceTarget(), target, c, n=0)
        for n in (2**32 + 3, 2**60, 2**63, 2**80):
            with pytest.raises(ValueError, match=r"^n is too large: at most 2147483647 runs$"):
                s.position_test(s.BacktestPriceTarget(), target, c, n=n)
        with pytest.raises(TypeError, match="n must be an integer, not bool"):
            s.position_test(s.BacktestPriceTarget(), target, c, n=True)
    assert s.position_test(s.BacktestPriceTarget(), tgt, c, n=np.int64(3)).null_stats["sharpe"].size == 3


def test_metric_and_engine_outputs_are_checked():
    ret, o, h, l, c = _market(100, 35)
    tgt = _held(np.sign(ret), 4)
    with pytest.raises(ValueError, match="one value per strategy in its input: 7 expected, got 1"):
        s.position_test(s.BacktestPriceTarget(), tgt, c, n=7, metric=lambda out: out[-1, 0, 0])
    with pytest.raises(ValueError, match="1 expected, got 2"):
        s.position_test(s.BacktestPortfolioTarget(), np.column_stack([tgt, -tgt]),
                        np.column_stack([c, c]), n=2, metric=lambda out: out[-1, :, 0])
    with pytest.raises(ValueError, match="2 expected, got 1"):
        s.walk_forward(s.BacktestPriceTarget(), np.column_stack([tgt, -tgt]), c, train=50,
                       test=10, metric=lambda out: 1.0)
    with pytest.raises(ValueError, match="engine must return a backtest output"):
        s.position_test(_Engine(out=np.zeros((100, 3))), tgt, c, n=2)
    with pytest.raises(ValueError, match="engine must return a backtest output"):
        s.walk_forward(_Engine(out=np.zeros((99, 4))), tgt, c, train=50, test=10)
