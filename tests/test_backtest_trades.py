import numpy as np
import pytest

import screamer as s
from screamer import backtest_report, backtest_trades

POS = [0, 1, 1, -1, -1, 0, 2, 2]
PNL = [0.0, -0.1, 0.5, 0.3, -0.2, 0.4, -0.1, 0.6]
COST = [0.0, 0.1, 0.0, 0.2, 0.0, 0.1, 0.1, 0.0]


def _hand():
    return np.column_stack([np.cumsum(PNL), PNL, POS, COST]).astype(float)


def test_hand_example():
    tr = backtest_trades(_hand())
    np.testing.assert_array_equal(tr["entry"], [1, 3, 6])
    np.testing.assert_array_equal(tr["exit"], [3, 5, 7])
    np.testing.assert_array_equal(tr["side"], [1, -1, 1])
    np.testing.assert_array_equal(tr["bars"], [2, 2, 1])
    np.testing.assert_allclose(tr["pnl"], [0.7, 0.2, 0.5], rtol=1e-12)    # the flip row's 0.3 closes trade 1
    np.testing.assert_allclose(tr["cost"], [0.3, 0.1, 0.1], rtol=1e-12)
    np.testing.assert_array_equal(tr["closed"], [True, True, False])
    assert tr["closed"].dtype == bool and tr["entry"].dtype == np.int64


def test_index_labels_entry_and_exit():
    tr = backtest_trades(_hand(), index=np.arange(8) * 10)
    np.testing.assert_array_equal(tr["entry"], [10, 30, 60])
    np.testing.assert_array_equal(tr["exit"], [30, 50, 70])
    np.testing.assert_array_equal(tr["bars"], [2, 2, 1])


def test_nan_rows_are_skipped_and_counted_in_bars():
    v = _hand()
    v[2] = np.nan
    tr = backtest_trades(v)
    np.testing.assert_array_equal(tr["entry"], [1, 3, 6])
    np.testing.assert_allclose(tr["pnl"], [0.2, 0.2, 0.5], rtol=1e-12)
    np.testing.assert_array_equal(tr["bars"], [2, 2, 1])


def test_resizing_stays_one_trade_and_no_position_gives_no_trades():
    v = np.column_stack([np.zeros(4), [0.0, 1, 1, 1], [0.0, 1, 3, 0], np.zeros(4)])
    tr = backtest_trades(v)
    np.testing.assert_array_equal(tr["entry"], [1])
    np.testing.assert_array_equal(tr["exit"], [3])
    empty = backtest_trades(np.zeros((5, 4)))
    assert all(len(a) == 0 for a in empty.values())


def _engine_outputs():
    rng = np.random.default_rng(0)
    T = 600
    price = 100 * np.exp(np.cumsum(rng.standard_normal(T) * 0.01))
    target = np.repeat(rng.integers(-2, 3, T // 20), 20).astype(float)
    o = np.r_[price[0], price[:-1]]
    h = np.maximum(o, price) * 1.002
    lo = np.minimum(o, price) * 0.998
    yield "price", s.BacktestPriceTarget(fee=1e-3)(target, price)
    yield "ohlc", s.BacktestOHLCTarget(taker_fee=5e-4)(target, o, h, lo, price)
    yield "ohlcv", s.BacktestOHLCVTarget()(target, o, h, lo, price, rng.uniform(1e3, 1e4, T))
    yield "trades", s.BacktestTradesTarget()(target, price, rng.uniform(1.0, 10.0, T))
    yield "l1", s.BacktestL1Target()(target, price * 0.9995, price * 1.0005, np.full(T, 5.0), np.full(T, 5.0))


@pytest.mark.parametrize("name,out", list(_engine_outputs()))
def test_closed_trades_agree_with_backtest_stats(name, out):
    out = np.asarray(out, dtype=float)
    tr = backtest_trades(out)
    _, summary = backtest_report(out)
    pnl, bars = tr["pnl"][tr["closed"]], tr["bars"][tr["closed"]]
    assert len(pnl) == summary["round_trips"] > 0
    wins = pnl > 0
    np.testing.assert_allclose(wins.mean(), summary["win_rate"], rtol=1e-12)
    np.testing.assert_allclose(pnl[wins].mean(), summary["avg_win"], rtol=1e-9)
    if (~wins).any():
        np.testing.assert_allclose(pnl[~wins].mean(), summary["avg_loss"], rtol=1e-9)
    np.testing.assert_allclose(bars.mean(), summary["avg_holding"], rtol=1e-12)


def test_portfolio_output_raises():
    with pytest.raises(ValueError, match="once per asset"):
        backtest_trades(np.zeros((10, 3, 4)))
    with pytest.raises(ValueError, match=r"\(T, 4\)"):
        backtest_trades(np.zeros((10, 3)))


def test_open_trade_with_nan_tail_rows_exits_on_the_last_row():
    v = _hand()
    v[-1] = np.nan                       # the open long's last row is NaN
    tr = backtest_trades(v)
    assert tr["exit"][-1] == len(v) - 1 and not tr["closed"][-1]
    np.testing.assert_allclose(tr["pnl"][-1], -0.1, rtol=1e-12)
    np.testing.assert_array_equal(tr["bars"][-1], len(v) - 1 - 6)


def test_close_on_a_nan_row_is_deferred_to_the_next_row():
    # the flat row 5 is NaN, so the short closes on row 6, which opens a long
    v = _hand()
    v[5] = np.nan
    tr = backtest_trades(v)
    np.testing.assert_array_equal(tr["entry"], [1, 3, 6])
    np.testing.assert_array_equal(tr["exit"], [3, 6, 7])
    np.testing.assert_array_equal(tr["side"], [1, -1, 1])
    # row 6 is a flip: its pnl and its cost go to the closing short
    np.testing.assert_allclose(tr["pnl"], [0.7, -0.3, 0.6], rtol=1e-12)
    np.testing.assert_allclose(tr["cost"], [0.3, 0.1, 0.0], atol=1e-15)
    np.testing.assert_array_equal(tr["closed"], [True, True, False])
