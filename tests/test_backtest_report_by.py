import numpy as np
import pytest

from screamer import BacktestPriceTarget, backtest_report


def _out(T=500, seed=0):
    rng = np.random.default_rng(seed)
    price = 100 + np.cumsum(rng.standard_normal(T))
    target = np.resize(np.repeat(rng.integers(-2, 3, T // 25 + 1), 25), T).astype(float)
    out = np.asarray(BacktestPriceTarget(fee=1e-3)(target, price), dtype=float)
    out[[40, 41]] = np.nan
    return out


def test_hand_example():
    pnl = np.array([1.0, -2.0, 3.0, 0.5, -1.0])
    v = np.column_stack([np.cumsum(pnl), pnl, [1.0, 1, 0, -1, -1], [0.1, 0, 0.1, 0.1, 0]])
    _, summary = backtest_report(v, by=np.array(["b", "b", "a", "b", "a"]))
    g = summary["by_group"]
    assert list(g) == ["b", "a"]
    b, a = g["b"], g["a"]
    np.testing.assert_allclose([b["total_pnl"], b["total_cost"], b["turnover"], b["max_drawdown"]],
                               [-0.5, 0.2, 2.0, -2.0], rtol=1e-12)
    np.testing.assert_allclose(b["sharpe"], np.mean([1, -2, 0.5]) / np.std([1, -2, 0.5], ddof=1), rtol=1e-12)
    np.testing.assert_allclose([a["total_pnl"], a["total_cost"], a["turnover"], a["max_drawdown"]],
                               [2.0, 0.1, 1.0, -1.0], rtol=1e-12)
    np.testing.assert_allclose(a["sharpe"], 1.0 / np.sqrt(8.0), rtol=1e-12)
    assert (b["steps"], a["steps"]) == (3, 2) and isinstance(b["steps"], int)


def test_group_totals_add_up_to_the_ungrouped_totals():
    out = _out()
    _, summary = backtest_report(out, by=np.arange(len(out)) // 100)
    g = summary["by_group"]
    assert list(g) == [0, 1, 2, 3, 4]
    for key in ("total_pnl", "total_cost", "turnover"):
        np.testing.assert_allclose(sum(v[key] for v in g.values()), summary[key], rtol=1e-9, atol=1e-9)
    assert sum(v["steps"] for v in g.values()) == int(np.sum(~np.isnan(out).any(axis=1)))


def test_a_single_label_equals_the_ungrouped_summary():
    out = _out(seed=1)
    _, summary = backtest_report(out, periods_per_year=252, by=np.full(len(out), "all"))
    g = summary["by_group"]["all"]
    for key in ("total_pnl", "total_cost", "turnover", "sharpe", "max_drawdown"):
        np.testing.assert_allclose(g[key], summary[key], rtol=1e-9, atol=1e-9)


def test_month_labels():
    out = _out(T=90)
    days = np.datetime64("2024-01-01") + np.arange(90)
    _, summary = backtest_report(out, by=days.astype("datetime64[M]"))
    assert list(summary["by_group"]) == [np.datetime64("2024-01"), np.datetime64("2024-02"),
                                         np.datetime64("2024-03")]
    assert [v["steps"] for v in summary["by_group"].values()] == [31, 29 - 2, 30]   # rows 40 and 41 are in February


def test_without_by_nothing_changes_and_bad_labels_raise():
    out = _out()
    running, summary = backtest_report(out)
    assert "by_group" not in summary
    running_by, summary_by = backtest_report(out, by=np.zeros(len(out)))
    np.testing.assert_equal({k: v for k, v in summary_by.items() if k != "by_group"}, summary)
    with pytest.raises(ValueError, match="one label per row"):
        backtest_report(out, by=np.zeros(len(out) - 1))
    with pytest.raises(ValueError, match="1-D"):
        backtest_report(out, by=np.zeros((len(out), 2)))


def test_mixed_type_object_labels():
    pnl = np.array([1.0, -2.0, 3.0, 0.5, -1.0])
    v = np.column_stack([np.cumsum(pnl), pnl, [1.0, 1, 0, -1, -1], [0.1, 0, 0.1, 0.1, 0]])
    _, mixed = backtest_report(v, by=np.array([1, 1, "a", 1, "a"], dtype=object))
    _, plain = backtest_report(v, by=np.array(["b", "b", "a", "b", "a"]))
    assert list(mixed["by_group"]) == [1, "a"]
    assert mixed["by_group"][1] == plain["by_group"]["b"]
    assert mixed["by_group"]["a"] == plain["by_group"]["a"]
    _, listed = backtest_report(v, by=[1, 1, "a", 1, "a"])     # a list becomes strings
    assert list(listed["by_group"]) == ["1", "a"]
