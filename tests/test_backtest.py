import numpy as np
from screamer import BacktestSignal, backtest_report


def test_frictionless_marks_to_market():
    # hold long 1 through 100 -> 102, flat at the end
    out = BacktestSignal()(np.array([1., 1, 0]), np.array([100., 101, 102]))
    np.testing.assert_allclose(out[:, 0], [0, 1, 2])       # equity
    np.testing.assert_allclose(out[:, 1], [0, 1, 1])       # pnl
    np.testing.assert_allclose(out[:, 2], [1, 1, 0])       # position
    np.testing.assert_allclose(out[:, 3], [0, 0, 0])       # cost


def test_frictionless_equals_cumsum_prev_position_times_dprice():
    rng = np.random.default_rng(0); n = 200
    price = 100 + np.cumsum(rng.standard_normal(n) * 0.3)
    signal = np.sign(rng.standard_normal(n))
    equity = BacktestSignal()(signal, price)[:, 0]
    prev_pos = np.concatenate([[0.0], signal[:-1]])
    dprice = np.concatenate([[0.0], np.diff(price)])
    np.testing.assert_allclose(equity, np.cumsum(prev_pos * dprice), atol=1e-9)


def test_taker_cost_is_turnover_times_half_spread():
    # long then flat at a flat price: each trade crosses half of a 1% spread
    out = BacktestSignal(spread=0.01)(np.array([1., 0]), np.array([100., 100]))
    np.testing.assert_allclose(out[:, 3], [0.5, 0.5])      # per-step cost
    np.testing.assert_allclose(out[-1, 0], -1.0)           # equity = -total cost


def test_fee_on_traded_notional():
    out = BacktestSignal(fee=0.001)(np.array([2., 2]), np.array([100., 100]))
    np.testing.assert_allclose(out[:, 3], [0.2, 0.0])      # 2 units * 100 * 0.001, then no trade


def test_short_position_pnl():
    # short 1 while price rises 100 -> 102 loses 2
    out = BacktestSignal()(np.array([-1., -1, 0]), np.array([100., 101, 102]))
    np.testing.assert_allclose(out[:, 0], [0, -1, -2])


def test_is_causal():
    price = np.array([100., 101, 102, 103]); signal = np.array([1., 1, -1, -1])
    full = BacktestSignal()(signal, price)
    trunc = BacktestSignal()(signal[:2], price[:2])
    np.testing.assert_allclose(full[:2], trunc)            # a future signal cannot change past rows


def test_stream_equals_batch():
    rng = np.random.default_rng(1); n = 200
    price = 100 + np.cumsum(rng.standard_normal(n) * 0.3)
    signal = np.sign(rng.standard_normal(n))
    op = BacktestSignal(spread=0.0005, fee=0.0002)
    stream = np.array([op(float(s), float(p)) for s, p in zip(signal, price)])
    batch = BacktestSignal(spread=0.0005, fee=0.0002)(signal, price)
    np.testing.assert_allclose(np.nan_to_num(stream), np.nan_to_num(batch))


def test_nan_skips_and_holds_position():
    out = BacktestSignal()(np.array([1., np.nan, 1.]), np.array([100., 101, 102]))
    assert np.all(np.isnan(out[1]))                        # NaN bar -> all-NaN row, state untouched
    # position held across the gap; t2 marks 1*(102-100)=2 against the last good price
    np.testing.assert_allclose(out[2], [2, 2, 1, 0])


def test_reset_restarts():
    price = np.array([100., 101, 100]); signal = np.array([1., 1, 1])
    op = BacktestSignal()
    a = op(signal, price); op.reset(); b = op(signal, price)
    np.testing.assert_allclose(a, b)


def test_backtest_report_shape_and_invariants():
    out = BacktestSignal(spread=0.01)(np.array([1., 1, 0]), np.array([100., 90, 95]))
    running, summary = backtest_report(out)
    assert list(summary.index) == [
        "total_pnl", "max_drawdown", "total_cost", "turnover", "num_trades", "sharpe"]
    assert summary["num_trades"] == 2.0                    # enter long, exit
    assert summary["turnover"] == 2.0                      # 1 unit in, 1 out
    assert summary["max_drawdown"] <= 0.0
    assert summary["total_pnl"] == running["equity"].ffill().iloc[-1]
    assert summary["total_cost"] == running["cost"].sum()
