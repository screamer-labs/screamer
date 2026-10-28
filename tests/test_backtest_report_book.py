"""backtest_report on the (T, K, 4) output of the portfolio engines: the book
summary, the per-asset summaries, and by= on the book."""
import numpy as np
import pytest

from screamer import BacktestPortfolioOHLCTarget, backtest_report

BOOK_KEYS = ["equity", "pnl", "cost",
             "drawdown", "cum_cost", "turnover", "trades", "max_drawdown", "sharpe",
             "exposure", "round_trips", "win_rate", "profit_factor", "avg_win",
             "avg_loss", "avg_holding", "sortino", "calmar"]


def _book(T=400, K=3, seed=0):
    rng = np.random.default_rng(seed)
    c = 100 * np.exp(np.cumsum(0.01 * rng.standard_normal((T, K)), axis=0))
    target = np.resize(np.repeat(rng.integers(-1, 2, (T // 20 + 1, K)), 20, axis=0), (T, K))
    target = 0.3 * target.astype(float)
    return np.asarray(BacktestPortfolioOHLCTarget(fee=1e-3)(target, c, c, c, c))


def _hand_sum(v):
    """The book summed asset by asset, in the order C++ sums it."""
    book = v[:, 0, :].copy()
    for k in range(1, v.shape[1]):
        book += v[:, k, :]
    return book


def _same(a, b):
    return (np.isnan(a) and np.isnan(b)) or a == b


def test_book_equals_the_hand_summed_book():
    v = _book()
    running, summary = backtest_report(v, periods_per_year=365)
    hand_running, hand = backtest_report(_hand_sum(v), periods_per_year=365)
    for key in ("total_pnl", "total_cost", "max_drawdown", "sharpe", "sortino", "calmar"):
        assert summary[key] == hand[key], key
    for key in ("equity", "pnl", "cost", "drawdown", "cum_cost", "max_drawdown", "sharpe",
                "sortino", "calmar"):
        np.testing.assert_array_equal(running[key], hand_running[key], err_msg=key)


def test_book_running_columns_and_summary_keys():
    v = _book()
    running, summary = backtest_report(v)
    _, one = backtest_report(v[:, 0, :])
    assert list(running) == BOOK_KEYS
    assert list(summary) == list(one) + ["per_asset"]
    assert all(a.shape == (len(v),) for a in running.values())
    assert summary["total_pnl"] == running["equity"][-1]


def test_per_asset_summaries_are_the_single_asset_reports():
    v = _book(K=4, seed=1)
    _, summary = backtest_report(v, periods_per_year=252)
    assert len(summary["per_asset"]) == 4
    for k, got in enumerate(summary["per_asset"]):
        _, want = backtest_report(v[:, k, :], periods_per_year=252)
        assert list(got) == list(want)
        for key in want:
            assert _same(got[key], want[key]), (k, key)
            assert isinstance(got[key], float)


def test_trades_and_exposure_count_over_all_assets():
    v = _book(K=3, seed=2)
    _, summary = backtest_report(v)
    per = summary["per_asset"]
    for key in ("turnover", "num_trades", "round_trips"):
        np.testing.assert_allclose(summary[key], sum(p[key] for p in per), rtol=1e-12)
    np.testing.assert_allclose(summary["exposure"], np.mean([p["exposure"] for p in per]),
                               rtol=1e-12)
    wins = sum(p["win_rate"] * p["round_trips"] for p in per)
    np.testing.assert_allclose(summary["win_rate"], wins / summary["round_trips"], rtol=1e-12)
    holding = sum(p["avg_holding"] * p["round_trips"] for p in per)
    np.testing.assert_allclose(summary["avg_holding"], holding / summary["round_trips"],
                               rtol=1e-12)


def test_one_asset_equals_the_single_asset_report():
    v = _book(K=1, seed=3)
    v[[10, 11]] = np.nan                       # skipped rows
    running, summary = backtest_report(v, periods_per_year=252)
    one_running, one = backtest_report(v[:, 0, :], periods_per_year=252)
    for key in one:
        assert _same(summary[key], one[key]), key
    for key in running:
        np.testing.assert_array_equal(running[key], one_running[key], err_msg=key)


def test_a_nan_leading_asset_joins_the_book_when_it_starts():
    v = _book(K=3, seed=4)
    v[:50, 2, :] = np.nan                      # asset 2 starts at row 50
    running, summary = backtest_report(v, periods_per_year=365)
    assert not np.isnan(running["equity"]).any()     # assets 0 and 1 carry the book
    _, hand = backtest_report(_hand_sum(np.nan_to_num(v, nan=0.0)), periods_per_year=365)
    for key in ("total_pnl", "total_cost", "max_drawdown", "sharpe", "sortino", "calmar"):
        np.testing.assert_allclose(summary[key], hand[key], rtol=1e-12, err_msg=key)
    _, late = backtest_report(v[:, 2, :], periods_per_year=365)
    for key in late:
        assert _same(summary["per_asset"][2][key], late[key]), key
    # Asset 2's leading rows do not count toward the exposure.
    per = summary["per_asset"]
    steps = [len(v), len(v), len(v) - 50]
    np.testing.assert_allclose(
        summary["exposure"],
        sum(p["exposure"] * n for p, n in zip(per, steps)) / sum(steps), rtol=1e-12)


def test_a_skipped_asset_holds_its_equity():
    v = _book(K=2, seed=5)
    v[100, 1, :] = np.nan
    running, _ = backtest_report(v)
    assert running["equity"][100] == v[100, 0, 0] + v[99, 1, 0]
    assert running["pnl"][100] == v[100, 0, 1]
    v[100, 0, :] = np.nan                      # every asset skipped: the row is skipped
    running, _ = backtest_report(v)
    assert all(np.isnan(running[k][100]) for k in running)


def test_by_groups_the_book():
    v = _book(K=3, seed=6)
    by = np.arange(len(v)) // 100
    _, summary = backtest_report(v, periods_per_year=365, by=by)
    hand_by = backtest_report(_hand_sum(v), periods_per_year=365, by=by)[1]["by_group"]
    g = summary["by_group"]
    assert list(g) == [0, 1, 2, 3]
    for label, row in g.items():
        for key in ("total_pnl", "total_cost", "sharpe", "max_drawdown", "steps"):
            assert row[key] == hand_by[label][key], (label, key)
    for key in ("total_pnl", "total_cost", "turnover"):
        np.testing.assert_allclose(sum(r[key] for r in g.values()), summary[key], rtol=1e-9)
    single = backtest_report(v, periods_per_year=365, by=np.zeros(len(v), int))[1]["by_group"][0]
    for key in ("total_pnl", "total_cost", "turnover", "sharpe", "max_drawdown"):
        np.testing.assert_allclose(single[key], summary[key], rtol=1e-12)


def test_layouts_and_lists():
    v = _book(T=60, K=2, seed=7)
    _, base = backtest_report(v)
    for layout in (np.asfortranarray(v), np.repeat(v, 2, axis=1)[:, ::2], v.tolist()):
        _, s2 = backtest_report(layout)
        for key in base:
            if key != "per_asset":
                assert _same(s2[key], base[key]), key
        for got, want in zip(s2["per_asset"], base["per_asset"]):
            assert all(_same(got[k], want[k]) for k in want)


def test_wrong_shapes_raise_naming_the_function():
    msg = (r"backtest_report: values must be a \(T, 4\) or \(T, K, 4\) array of "
           r"\[equity, pnl, position, cost\]")
    for bad in (np.zeros((3, 3)), np.zeros(4), np.zeros((2, 3, 3)), np.zeros((2, 1, 1, 4)),
                5.0, []):
        with pytest.raises(ValueError, match=msg):
            backtest_report(bad)
        with pytest.raises(ValueError, match=msg):
            backtest_report(bad, by=np.zeros(3))
    with pytest.raises(ValueError, match="backtest_report: .*at least one asset"):
        backtest_report(np.zeros((3, 0, 4)))
    with pytest.raises(ValueError, match="one label per row"):
        backtest_report(np.zeros((3, 2, 4)), by=[0, 1])


def test_periods_per_year_is_checked_for_a_book():
    v = np.zeros((3, 2, 4))
    for ppy in (0, -1, np.nan, np.inf):
        with pytest.raises(ValueError, match="periods_per_year must be finite and positive"):
            backtest_report(v, periods_per_year=ppy)
    with pytest.raises(TypeError, match="periods_per_year is a str, expected a number"):
        backtest_report(v, periods_per_year="252")
    _, summary = backtest_report(np.zeros((0, 2, 4)))
    assert np.isnan(summary["total_pnl"]) and len(summary["per_asset"]) == 2
