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
        s.position_test(eng, tgt, o, h, l, c, metric="sortino")
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
    d = s.permute_returns(spread, 5, log=False, seed=2)
    np.testing.assert_allclose(np.sort(np.diff(d[:, 0])), np.sort(np.diff(spread)), atol=1e-12)
    with pytest.raises(ValueError):
        s.permute_returns(spread, 5)                                 # log needs positive values


def test_permutations_seeded_and_chunk_invariant():
    from screamer.screamer_bindings import _permute_bars, _permute_returns
    o, h, l, c = _ohlc_walk(60, 4)
    full = _permute_bars(o, h, l, c, 9, 0, 12)
    parts = [_permute_bars(o, h, l, c, 9, a, b) for a, b in ((0, 5), (5, 7))]
    for k in range(5):
        np.testing.assert_array_equal(full[k], np.hstack([parts[0][k], parts[1][k]]))
    r = _permute_returns(c, True, 9, 0, 8)
    r2 = [_permute_returns(c, True, 9, a, b) for a, b in ((0, 3), (3, 5))]
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
