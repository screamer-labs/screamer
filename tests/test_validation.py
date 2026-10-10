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
