"""Tests for walk_forward: fold layout, selection, stitching, and statistics."""
import numpy as np
import pytest

from screamer import (BacktestCarry, BacktestL1Target, BacktestOHLCTarget, BacktestOHLCVTarget,
                      BacktestPortfolioTarget, BacktestPriceTarget, BacktestTradesTarget,
                      walk_forward)


def _market(T):
    """The deterministic OHLC market of js/test/validation.test.ts."""
    o, h, l, c = [], [], [], []
    p = 100.0
    for t in range(T):
        prev = p
        p *= np.exp(0.01 * np.sin(1.7 * t) * np.cos(0.31 * t * t))
        o.append(prev); c.append(p)
        h.append(max(prev, p) * 1.001); l.append(min(prev, p) * 0.999)
    return tuple(np.array(x) for x in (o, h, l, c))


def _targets(T):
    """Three deterministic candidates with different holding periods."""
    t = np.arange(T)
    return np.column_stack([np.sign(np.sin(t / 6.0)), np.sign(np.cos(t / 17.0)),
                            ((t // 5) % 3) - 1.0]).astype(float)


def _brute(engine, targets, prices, a, b, ppy=1.0):
    """Sharpe, total pnl, and max drawdown of rows [a, b) of each candidate's own run."""
    out = []
    for k in range(targets.shape[1]):
        engine.reset()
        o = np.asarray(engine(np.ascontiguousarray(targets[:, k]), *prices))
        pnl = o[a:b, 1]
        pnl = pnl[~np.isnan(pnl)]
        eq = np.cumsum(pnl)
        sharpe = pnl.mean() / pnl.std(ddof=1) * np.sqrt(ppy) if pnl.std(ddof=1) > 0 else np.nan
        out.append((sharpe, eq[-1], np.min(eq - np.maximum.accumulate(eq))))
    engine.reset()
    return np.array(out)


def test_fold_layout():
    o, h, l, c = _market(310)
    res = walk_forward(BacktestOHLCTarget(), _targets(310), o, h, l, c, train=100, test=50)
    assert res.fold_start.tolist() == [100, 150, 200, 250, 300]
    assert res.train_scores.shape == (5, 3)
    assert res.choice.shape == (5,)
    assert res.target.shape == (310,)
    assert res.output.shape == (310, 4)
    assert res.metric == "sharpe"


@pytest.mark.parametrize("metric,col,larger", [("sharpe", 0, True), ("total_pnl", 1, True),
                                               ("max_drawdown", 2, True)])
@pytest.mark.parametrize("anchored", [False, True])
def test_scores_and_choice_match_brute_force(metric, col, larger, anchored):
    T = 400
    o, h, l, c = _market(T)
    X = _targets(T)
    eng = BacktestOHLCTarget(taker_fee=2e-4)
    res = walk_forward(eng, X, o, h, l, c, train=120, test=40, metric=metric,
                       anchored=anchored, periods_per_year=252)
    for i, t0 in enumerate(res.fold_start):
        ref = _brute(eng, X, (o, h, l, c), 0 if anchored else t0 - 120, t0, ppy=252)[:, col]
        np.testing.assert_allclose(res.train_scores[i], ref, rtol=1e-9, atol=1e-12)
        assert res.choice[i] == int(np.argmax(ref if larger else -ref))


def test_anchored_and_rolling_differ_after_first_fold():
    T = 400
    o, h, l, c = _market(T)
    X = _targets(T)
    roll = walk_forward(BacktestOHLCTarget(), X, o, h, l, c, train=120, test=40)
    anch = walk_forward(BacktestOHLCTarget(), X, o, h, l, c, train=120, test=40, anchored=True)
    np.testing.assert_array_equal(roll.train_scores[0], anch.train_scores[0])
    assert not np.allclose(roll.train_scores[1:], anch.train_scores[1:])


def test_smaller_is_better_metric():
    T = 300
    o, h, l, c = _market(T)
    X = _targets(T)
    res = walk_forward(BacktestOHLCTarget(taker_fee=1e-3), X, o, h, l, c, train=100, test=50,
                       metric="total_cost")
    for i in range(res.choice.size):
        assert res.choice[i] == int(np.argmin(res.train_scores[i]))
    flipped = walk_forward(BacktestOHLCTarget(taker_fee=1e-3), X, o, h, l, c, train=100,
                           test=50, metric="total_cost", larger_is_better=True)
    for i in range(flipped.choice.size):
        assert flipped.choice[i] == int(np.argmax(flipped.train_scores[i]))


def test_ties_go_to_the_lowest_column():
    T = 300
    o, h, l, c = _market(T)
    X = _targets(T)
    dup = np.column_stack([X[:, 2], X[:, 0], X[:, 0], X[:, 2]])
    res = walk_forward(BacktestOHLCTarget(), dup, o, h, l, c, train=100, test=50)
    for i in range(res.choice.size):
        s = res.train_scores[i]
        assert s[1] == s[2] and s[0] == s[3]
        assert res.choice[i] == (1 if s[1] > s[0] else 0)


def test_nan_scores_lose_and_all_nan_stays_flat():
    T = 300
    o, h, l, c = _market(T)
    X = _targets(T)
    nan_col = np.full(T, np.nan)
    res = walk_forward(BacktestOHLCTarget(), np.column_stack([nan_col, X[:, 0]]),
                       o, h, l, c, train=100, test=50)
    assert np.isnan(res.train_scores[:, 0]).all()
    assert (res.choice == 1).all()

    # Every candidate is in its NaN warmup through the first two training windows.
    late = X.copy()
    late[:150] = np.nan
    res = walk_forward(BacktestOHLCTarget(), late, o, h, l, c, train=100, test=50)
    assert res.choice.tolist()[:2] == [-1, -1]
    assert (res.target[100:200] == 0).all()
    assert (res.choice[2:] >= 0).all()


def test_stitched_target_output_and_stats():
    T = 330
    o, h, l, c = _market(T)
    X = _targets(T)
    eng = BacktestOHLCTarget(taker_fee=5e-4)
    res = walk_forward(eng, X, o, h, l, c, train=100, test=60, periods_per_year=252)
    expect = np.zeros(T)
    for t0, k in zip(res.fold_start, res.choice):
        expect[t0:t0 + 60] = X[t0:t0 + 60, k]
    np.testing.assert_array_equal(res.target, expect)

    eng.reset()
    np.testing.assert_array_equal(res.output, np.asarray(eng(expect, o, h, l, c)))
    pnl = res.output[100:, 1]
    pnl = pnl[~np.isnan(pnl)]
    assert res.stats["total_pnl"] == pytest.approx(pnl.sum(), rel=1e-12, abs=1e-12)
    assert res.stats["sharpe"] == pytest.approx(pnl.mean() / pnl.std(ddof=1) * np.sqrt(252),
                                                rel=1e-9)
    assert res.stats["total_cost"] == pytest.approx(np.nansum(res.output[100:, 3]), rel=1e-12)
    assert set(res.stats) == {
        "total_pnl", "max_drawdown", "total_cost", "turnover", "num_trades", "sharpe",
        "exposure", "round_trips", "win_rate", "profit_factor", "avg_win", "avg_loss",
        "avg_holding", "sortino", "calmar"}


def test_one_candidate_and_price_engine():
    T = 200
    _, _, _, c = _market(T)
    x = _targets(T)[:, 0]
    res = walk_forward(BacktestPriceTarget(fee=1e-4), x, c, train=80, test=30)
    assert (res.choice == 0).all()
    np.testing.assert_array_equal(res.target[80:], x[80:])
    assert (res.target[:80] == 0).all()


def test_callable_metric():
    T = 300
    o, h, l, c = _market(T)
    X = _targets(T)

    def last_pnl_sum(out):                   # (L, k, 4) -> k values
        return np.nansum(out[:, :, 1], axis=0)

    res = walk_forward(BacktestOHLCTarget(), X, o, h, l, c, train=100, test=50,
                       metric=last_pnl_sum)
    ref = walk_forward(BacktestOHLCTarget(), X, o, h, l, c, train=100, test=50,
                       metric="total_pnl")
    np.testing.assert_allclose(res.train_scores, ref.train_scores, rtol=1e-9, atol=1e-12)
    assert res.metric == "last_pnl_sum"


def test_input_errors():
    T = 100
    o, h, l, c = _market(T)
    X = _targets(T)
    eng = BacktestOHLCTarget()
    with pytest.raises(ValueError, match="train"):
        walk_forward(eng, X, o, h, l, c, train=1, test=10)
    with pytest.raises(ValueError, match="test"):
        walk_forward(eng, X, o, h, l, c, train=10, test=0)
    with pytest.raises(ValueError, match="shorter"):
        walk_forward(eng, X, o, h, l, c, train=100, test=10)
    with pytest.raises(ValueError, match="metric"):
        walk_forward(eng, X, o, h, l, c, train=10, test=10, metric="bogus")
    with pytest.raises(ValueError, match="direction"):
        walk_forward(eng, X, o, h, l, c, train=10, test=10, metric="turnover")
    with pytest.raises(ValueError, match="same length"):
        walk_forward(eng, X, o[:-1], h, l, c, train=10, test=10)
    with pytest.raises(ValueError, match=r"\(T, K\) or \(T, K, C\)"):
        walk_forward(BacktestPortfolioTarget(), X[:, 0], np.ones((T, 3)), train=10, test=10)


def test_golden_values():
    # Same inputs and values as js/test/walk_forward.test.ts.
    T = 260
    o, h, l, c = _market(T)
    res = walk_forward(BacktestOHLCTarget(taker_fee=2e-4), _targets(T), o, h, l, c,
                       train=90, test=40, periods_per_year=252)
    assert res.fold_start.tolist() == [90, 130, 170, 210, 250]
    assert res.choice.tolist() == GOLDEN_CHOICE
    np.testing.assert_allclose(res.train_scores[:, 0], GOLDEN_SCORES_COL0, rtol=1e-12)
    assert res.stats["sharpe"] == pytest.approx(GOLDEN_SHARPE, rel=1e-12)
    assert res.stats["total_pnl"] == pytest.approx(GOLDEN_PNL, rel=1e-12)
    anch = walk_forward(BacktestOHLCTarget(taker_fee=2e-4), _targets(T), o, h, l, c,
                        train=90, test=40, anchored=True, metric="total_pnl")
    assert anch.choice.tolist() == GOLDEN_ANCHORED_CHOICE


GOLDEN_CHOICE = [1, 1, 0, 2, 2]
GOLDEN_SCORES_COL0 = [1.2084732360973462, 0.9425475505694672, 1.620956872474087,
                      -1.0708210557093047, -2.150979534091671]
GOLDEN_SHARPE = -0.8521351650503325
GOLDEN_PNL = -4.302590266628476
GOLDEN_ANCHORED_CHOICE = [1, 1, 1, 0, 0]


def _rate(T):
    return 1e-3 * np.sin(np.arange(T) / 7.0)


def test_zero_carry_equals_no_carry():
    T = 300
    o, h, l, c = _market(T)
    X = _targets(T)
    a = walk_forward(BacktestOHLCTarget(taker_fee=2e-4), X, o, h, l, c, train=100, test=50)
    b = walk_forward(BacktestOHLCTarget(taker_fee=2e-4), X, o, h, l, c, train=100, test=50,
                     carry=np.zeros(T))
    np.testing.assert_array_equal(a.train_scores, b.train_scores)
    np.testing.assert_array_equal(a.choice, b.choice)
    np.testing.assert_array_equal(a.output, b.output)
    np.testing.assert_equal(a.stats, b.stats)


def test_carry_is_backtest_carry_on_the_close():
    T = 300
    o, h, l, c = _market(T)
    X = _targets(T)
    rate = _rate(T)
    res = walk_forward(BacktestOHLCTarget(taker_fee=2e-4), X, o, h, l, c, train=100, test=50,
                       metric="total_pnl", carry=rate, carry_multiplier=2.0)
    eng = BacktestOHLCTarget(taker_fee=2e-4)
    raw = np.asarray(eng(res.target, o, h, l, c))
    want = np.asarray(BacktestCarry(multiplier=2.0)(*raw.T, c, rate))
    np.testing.assert_allclose(res.output, want, rtol=1e-12, atol=1e-12)
    for k in range(X.shape[1]):
        eng.reset()
        run = np.asarray(eng(np.ascontiguousarray(X[:, k]), o, h, l, c))
        carried = np.asarray(BacktestCarry(multiplier=2.0)(*run.T, c, rate))
        for i, t0 in enumerate(res.fold_start):
            np.testing.assert_allclose(res.train_scores[i, k], np.nansum(carried[t0 - 100:t0, 1]),
                                       rtol=1e-9, atol=1e-12)
    assert res.stats["total_pnl"] == pytest.approx(np.nansum(res.output[100:, 1]), rel=1e-12, abs=1e-12)


@pytest.mark.parametrize("kind", ["ohlcv", "trades", "l1"])
def test_carry_marks_the_last_price_input(kind):
    T = 300
    o, h, l, c = _market(T)
    X = _targets(T)
    rate = _rate(T)
    if kind == "ohlcv":
        make, market, mark = BacktestOHLCVTarget, (o, h, l, c, np.full(T, 1e6)), c
    elif kind == "trades":
        make, market, mark = BacktestTradesTarget, (c, np.full(T, 100.0)), c
    else:
        bid, ask = c * 0.999, c * 1.001
        make, market, mark = BacktestL1Target, (bid, ask, np.full(T, 10.0), np.full(T, 10.0)), ask
    res = walk_forward(make(), X, *market, train=100, test=50, carry=rate)
    raw = np.asarray(make()(res.target, *market))
    np.testing.assert_allclose(res.output, np.asarray(BacktestCarry()(*raw.T, mark, rate)),
                               rtol=1e-12, atol=1e-12)


def test_carry_argument_checks():
    T = 200
    o, h, l, c = _market(T)
    X = _targets(T)
    with pytest.raises(ValueError, match=r"carry must be a \(T,\) series"):
        walk_forward(BacktestOHLCTarget(), X, o, h, l, c, train=50, test=50, carry=np.zeros(T - 1))
    with pytest.raises(ValueError, match="carry_multiplier"):
        walk_forward(BacktestOHLCTarget(), X, o, h, l, c, train=50, test=50, carry=np.zeros(T),
                     carry_multiplier=0.0)


def test_carry_with_an_engine_without_a_mark_input_raises():
    class Wrapped:
        def __init__(self):
            self.eng = BacktestOHLCTarget()

        def __call__(self, target, *market):
            return self.eng(target, *market)

        def reset(self):
            self.eng.reset()

    T = 200
    o, h, l, c = _market(T)
    X = _targets(T)
    walk_forward(Wrapped(), X, o, h, l, c, train=50, test=50)
    with pytest.raises(ValueError, match="carry needs an engine with a mark input.*Wrapped has none"):
        walk_forward(Wrapped(), X, o, h, l, c, train=50, test=50, carry=np.zeros(T))
    # declaring the mark input is enough
    Wrapped.mark_input = 3
    rate = _rate(T)
    np.testing.assert_array_equal(
        walk_forward(Wrapped(), X, o, h, l, c, train=50, test=50, carry=rate).output,
        walk_forward(BacktestOHLCTarget(), X, o, h, l, c, train=50, test=50, carry=rate).output)


def test_engines_declare_their_mark_input():
    from screamer import BacktestPortfolioOHLCTarget
    marks = {BacktestPriceTarget: 0, BacktestOHLCTarget: 3, BacktestOHLCVTarget: 3,
             BacktestTradesTarget: 0, BacktestL1Target: 1, BacktestPortfolioTarget: 0,
             BacktestPortfolioOHLCTarget: 3}
    for engine, mark in marks.items():
        assert engine.mark_input == mark and engine().mark_input == mark, engine.__name__
    assert not hasattr(BacktestCarry(), "mark_input")


def test_carry_with_a_subclass_of_an_engine():
    class MyOHLC(BacktestOHLCTarget):
        pass

    T = 200
    o, h, l, c = _market(T)
    X = _targets(T)
    rate = _rate(T)
    a = walk_forward(MyOHLC(taker_fee=2e-4), X, o, h, l, c, train=50, test=50, carry=rate)
    b = walk_forward(BacktestOHLCTarget(taker_fee=2e-4), X, o, h, l, c, train=50, test=50, carry=rate)
    np.testing.assert_array_equal(a.output, b.output)
    np.testing.assert_array_equal(a.train_scores, b.train_scores)


def _book(T, K, seed):
    rng = np.random.default_rng(seed)
    return 100 * np.exp(np.cumsum(rng.standard_normal((T, K)) * 0.01, axis=0))


def _book_targets(T, K, C):
    t = np.arange(T)[:, None, None]
    return 0.3 * np.sign(np.sin(t / (5.0 + 3.0 * np.arange(C))[None, None, :]
                                + np.arange(K)[None, :, None]))


def _book_sharpe(out, a, b):
    pnl = np.nansum(out[a:b, :, 1], axis=1)
    return pnl.mean() / pnl.std(ddof=1)


def test_portfolio_scores_selection_and_stitching():
    T, K, C = 300, 3, 4
    prices, X = _book(T, K, 0), _book_targets(T, K, C)
    eng = BacktestPortfolioTarget(fee=1e-4)
    res = walk_forward(eng, X, prices, train=100, test=50)
    assert res.train_scores.shape == (4, C)
    assert res.target.shape == (T, K) and res.output.shape == (T, K, 4)
    for c in range(C):
        eng.reset()
        out = np.asarray(eng(np.ascontiguousarray(X[:, :, c]), prices))
        for i, t0 in enumerate(res.fold_start):
            np.testing.assert_allclose(res.train_scores[i, c], _book_sharpe(out, t0 - 100, t0), rtol=1e-9)
    expect = np.zeros((T, K))
    for t0, c in zip(res.fold_start, res.choice):
        expect[t0:t0 + 50] = X[t0:t0 + 50, :, c]
    np.testing.assert_array_equal(res.target, expect)
    eng.reset()
    np.testing.assert_array_equal(res.output, np.asarray(eng(expect, prices)))


def test_portfolio_with_one_asset_equals_the_single_asset_result():
    T = 300
    _, _, _, c = _market(T)
    X = _targets(T)
    book = lambda: BacktestPortfolioTarget(fee=1e-4, size="units", initial_capital=0.0)  # noqa: E731
    # the two engines agree on one asset
    np.testing.assert_allclose(np.asarray(book()(X[:, :1], c[:, None]))[:, 0, :],
                               np.asarray(BacktestPriceTarget(fee=1e-4)(X[:, 0], c)), rtol=1e-12, atol=1e-12)
    a = walk_forward(BacktestPriceTarget(fee=1e-4), X, c, train=100, test=50)
    b = walk_forward(book(), X[:, None, :], c[:, None], train=100, test=50)
    np.testing.assert_allclose(b.train_scores, a.train_scores, rtol=1e-9, atol=1e-12)
    np.testing.assert_array_equal(b.choice, a.choice)
    np.testing.assert_array_equal(b.target[:, 0], a.target)
    np.testing.assert_allclose(b.output[:, 0, :], a.output, rtol=1e-12, atol=1e-12)
    for key in ("total_pnl", "total_cost", "turnover", "num_trades", "sharpe", "max_drawdown"):
        np.testing.assert_allclose(b.stats[key], a.stats[key], rtol=1e-9, atol=1e-12)


def test_portfolio_callable_metric_gets_one_candidate_book():
    T, K, C = 240, 2, 3
    prices, X = _book(T, K, 1), _book_targets(T, K, C)
    shapes = []

    def book_pnl(out):
        shapes.append(out.shape)
        return float(np.nansum(out[:, :, 1]))

    res = walk_forward(BacktestPortfolioTarget(), X, prices, train=80, test=40, metric=book_pnl)
    assert shapes and all(sh[1:] == (K, 4) for sh in shapes)
    eng = BacktestPortfolioTarget()
    for c in range(C):
        eng.reset()
        out = np.asarray(eng(np.ascontiguousarray(X[:, :, c]), prices))
        for i, t0 in enumerate(res.fold_start):
            np.testing.assert_allclose(res.train_scores[i, c], np.nansum(out[t0 - 80:t0, :, 1]),
                                       rtol=1e-9, atol=1e-12)


def test_portfolio_carry_per_asset():
    T, K, C = 240, 3, 2
    prices, X = _book(T, K, 2), _book_targets(T, K, C)
    rate = 1e-3 * np.cos(np.arange(T)[:, None] / 5.0 + np.arange(K))
    res = walk_forward(BacktestPortfolioTarget(), X, prices, train=80, test=40, carry=rate)
    raw = np.asarray(BacktestPortfolioTarget()(res.target, prices))
    want = np.stack([np.asarray(BacktestCarry()(*raw[:, k, :].T, prices[:, k], rate[:, k]))
                     for k in range(K)], axis=1)
    np.testing.assert_allclose(res.output, want, rtol=1e-12, atol=1e-12)


def test_portfolio_argument_checks():
    T = 120
    prices = _book(T, 3, 3)
    with pytest.raises(ValueError, match=r"\(T, K\) or \(T, K, C\)"):
        walk_forward(BacktestPortfolioTarget(), np.ones(T), prices, train=40, test=40)
    with pytest.raises(ValueError, match="price inputs"):
        walk_forward(BacktestPortfolioTarget(), np.ones((T, 3, 2)), prices[:, :2], train=40, test=40)
    with pytest.raises(ValueError, match=r"carry must be a \(T, K\) array"):
        walk_forward(BacktestPortfolioTarget(), np.ones((T, 3, 2)), prices, train=40, test=40,
                     carry=np.zeros(T))


def test_portfolio_restates_a_held_target_when_the_book_changes():
    # Rebalancing schedules: orders every 7 or 11 steps, NaN (hold) between.
    T, K, C = 260, 2, 2
    prices = _book(T, K, 4)
    X = np.full((T, K, C), np.nan)
    X[::7, :, 0] = _book_targets(T, K, 1)[::7, :, 0]
    X[::11, :, 1] = -_book_targets(T, K, 1)[::11, :, 0]
    X[::11, 1, 1] = np.nan                    # asset 1 of candidate 1 never trades
    res = walk_forward(BacktestPortfolioTarget(), X, prices, train=80, test=40)
    expect = np.zeros((T, K))
    assert (res.choice >= 0).all()
    assert len(set(res.choice.tolist())) > 1 and (res.choice[1:] == res.choice[:-1]).any()
    prev = -1
    for t0, c in zip(res.fold_start, res.choice):
        expect[t0:t0 + 40] = X[t0:t0 + 40, :, c]
        for k in range(K):
            if c != prev and np.isnan(expect[t0, k]):   # a new book: the order c stands on at t0
                past = X[:t0, k, c][~np.isnan(X[:t0, k, c])]
                expect[t0, k] = past[-1] if past.size else 0.0
        prev = c
    np.testing.assert_array_equal(res.target, expect)
    switched = np.r_[True, res.choice[1:] != res.choice[:-1]]
    assert not np.isnan(res.target[res.fold_start[switched]]).any()
    assert np.isnan(res.target[res.fold_start[~switched]]).any()


def test_portfolio_continuing_candidate_carries_its_book_without_a_trade():
    # One candidate on a sparse schedule: orders at 5, 35, 65, ..., never on a fold start.
    T, K = 300, 2
    prices = _book(T, K, 5)
    X = np.full((T, K, 1), np.nan)
    X[5::30, :, 0] = _book_targets(T, K, 1)[5::30, :, 0]
    for size in ("weight", "units"):
        make = lambda: BacktestPortfolioTarget(fee=1e-3, size=size)  # noqa: E731
        res = walk_forward(make(), X, prices, train=80, test=40)
        assert (res.choice == 0).all()
        expect = np.zeros((T, K))
        expect[80:] = X[80:, :, 0]
        expect[80] = X[65, :, 0]              # restated once, at the first test window
        np.testing.assert_array_equal(res.target, expect)
        np.testing.assert_array_equal(res.output, np.asarray(make()(expect, prices)))
        later = res.fold_start[1:]
        assert (res.output[later, :, 3] == 0).all()                      # no cost at later fold starts
        np.testing.assert_array_equal(res.output[later, :, 2], res.output[later - 1, :, 2])
        if size == "units":
            # after the first test row the stitched book is the candidate's own run
            own = np.asarray(make()(np.ascontiguousarray(X[:, :, 0]), prices))
            np.testing.assert_allclose(res.output[81:, :, 1:3], own[81:, :, 1:3], rtol=1e-12, atol=1e-12)
            np.testing.assert_array_equal(res.output[81:, :, 3], own[81:, :, 3])
