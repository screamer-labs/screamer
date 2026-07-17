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


# --- BacktestOHLC ------------------------------------------------------------

def test_ohlc_market_order_marks_to_close():
    from screamer import BacktestOHLC
    o = BacktestOHLC()(np.array([1., 1, 0]), np.array([np.nan] * 3),
                       np.array([100., 101, 102]), np.array([100., 101, 103]),
                       np.array([99., 100, 101]), np.array([100., 101, 102]))
    np.testing.assert_allclose(o[:, 0], [0, 1, 2])       # equity: long 1 from 100 to 102
    np.testing.assert_allclose(o[:, 2], [1, 1, 0])       # position


def test_ohlc_limit_fills_only_when_range_reaches():
    from screamer import BacktestOHLC
    # buy limit 99: the bar low 98 reaches it -> fill; next bar low 99.5 does not
    reached = BacktestOHLC()(np.array([1.]), np.array([99.]), np.array([100.]),
                             np.array([101.]), np.array([98.]), np.array([100.]))
    assert reached[0, 2] == 1.0
    missed = BacktestOHLC()(np.array([1.]), np.array([99.]), np.array([100.]),
                            np.array([101.]), np.array([99.5]), np.array([100.]))
    assert missed[0, 2] == 0.0                           # not reached -> no fill
    # breach is stricter than touch at the exact level
    touch = BacktestOHLC(fill="touch")(np.array([1.]), np.array([99.]), np.array([100.]),
                                       np.array([101.]), np.array([99.]), np.array([100.]))
    breach = BacktestOHLC(fill="breach")(np.array([1.]), np.array([99.]), np.array([100.]),
                                         np.array([101.]), np.array([99.]), np.array([100.]))
    assert touch[0, 2] == 1.0 and breach[0, 2] == 0.0


def test_ohlc_stream_equals_batch_and_reset():
    from screamer import BacktestOHLC
    rng = np.random.default_rng(0); n = 100
    c = 100 + np.cumsum(rng.standard_normal(n) * 0.3)
    o, h, l = c - 0.1, c + 0.5, c - 0.5
    tgt = np.sign(rng.standard_normal(n)); lim = np.full(n, np.nan)
    op = BacktestOHLC(spread=0.001)
    stream = np.array([op(float(tgt[i]), lim[i], float(o[i]), float(h[i]), float(l[i]), float(c[i]))
                       for i in range(n)])
    op.reset()
    batch = BacktestOHLC(spread=0.001)(tgt, lim, o, h, l, c)
    np.testing.assert_allclose(np.nan_to_num(stream), np.nan_to_num(batch))


# --- BacktestTrades ----------------------------------------------------------

def test_trades_fill_and_adverse_selection():
    from screamer import BacktestTrades
    # resting buy 1 @ 100; a print at 99 (<=100) size 2 fills 1 @ 100, marks at 99 (down 1);
    # then a print at 101 with the order still resting (101 > 100, no fill) marks the position up
    t = BacktestTrades()(np.array([100., 100]), np.array([1., 1]),
                         np.array([99., 101]), np.array([2., 5]))
    np.testing.assert_allclose(t[:, 2], [1, 1])          # position filled then held
    np.testing.assert_allclose(t[0], [-1, -1, 1, 1])     # bought at 100 vs 99 mark -> cost 1
    np.testing.assert_allclose(t[1, 0], 1.0)             # mark 99 -> 101 recovers to +1


def test_trades_partial_up_to_print_size():
    from screamer import BacktestTrades
    # resting buy 5 @ 100; a print AT 100 size 2 fills min(5, 1.0*2)=2 (participation 1.0)
    t = BacktestTrades()(np.array([100.]), np.array([5.]), np.array([100.]), np.array([2.]))
    assert t[0, 2] == 2.0


def test_trades_through_fills_full_order_not_print_size():
    from screamer import BacktestTrades
    # resting buy 10 @ 100; a print of size 2 at 99 trades THROUGH -> full 10 fills
    t = BacktestTrades()(np.array([100.]), np.array([10.]), np.array([99.]), np.array([2.]))
    assert t[0, 2] == 10.0                                  # swept: full order, not min(10,2)


def test_trades_at_fills_participation_of_trade_size():
    from screamer import BacktestTrades
    # resting buy 10 @ 100; a print of size 8 AT 100, participation 0.5 -> min(10, 0.5*8)=4
    t = BacktestTrades(participation_ratio=0.5)(
        np.array([100.]), np.array([10.]), np.array([100.]), np.array([8.]))
    assert t[0, 2] == 4.0


def test_trades_participation_capped_by_order_no_zeno():
    from screamer import BacktestTrades
    # participation*trade_size exceeds the order -> full fill, capped by remaining
    t = BacktestTrades(participation_ratio=0.5)(
        np.array([100.]), np.array([10.]), np.array([100.]), np.array([100.]))
    assert t[0, 2] == 10.0                                  # min(10, 0.5*100)=10, no dust


def test_trades_breach_ignores_at_price():
    from screamer import BacktestTrades
    # breach: a print exactly AT 100 does not fill; only strictly through does
    at = BacktestTrades(fill="breach")(np.array([100.]), np.array([10.]),
                                       np.array([100.]), np.array([8.]))
    assert at[0, 2] == 0.0
    through = BacktestTrades(fill="breach")(np.array([100.]), np.array([10.]),
                                            np.array([99.]), np.array([2.]))
    assert through[0, 2] == 10.0


def test_trades_stream_equals_batch():
    from screamer import BacktestTrades
    rng = np.random.default_rng(1); n = 100
    price = 100 + np.cumsum(rng.standard_normal(n) * 0.2)
    op = np.where(rng.standard_normal(n) > 0, price - 0.1, price + 0.1)   # order near the print
    os_ = rng.choice([-1.0, 1.0], n)
    ts = np.abs(rng.standard_normal(n)) + 1
    eng = BacktestTrades(maker_fee=-0.0001)
    stream = np.array([eng(float(op[i]), float(os_[i]), float(price[i]), float(ts[i])) for i in range(n)])
    batch = BacktestTrades(maker_fee=-0.0001)(op, os_, price, ts)
    np.testing.assert_allclose(np.nan_to_num(stream), np.nan_to_num(batch))


# --- BacktestL1 --------------------------------------------------------------

def test_l1_two_sided_fills_and_spread_capture():
    from screamer import BacktestL1
    # passive quotes at the market touch: buy the bid, sell the ask, earn the spread.
    bid = np.array([99.5, 99.5]); ask = np.array([100.5, 100.5])
    bsz = np.array([5., 5]); asz = np.array([5., 5])
    # our bid at the market bid; the market ask reaches it only when ask <= 99.5 (never here)
    # instead quote a bid the market crosses into: my_bid 100.5 (>= ask) fills a buy at 100.5
    out = BacktestL1()(bid, ask, bsz, asz,
                       np.array([100.5, 100.5]), np.array([1., 0.]),   # buy quote then withdrawn
                       np.array([np.nan, 99.5]), np.array([np.nan, 1.]))  # then a sell quote
    assert out[0, 2] == 1.0                              # bought
    assert out[1, 2] == 0.0                              # sold back (market bid 99.5 >= my_ask 99.5)


def test_l1_inventory_cap():
    from screamer import BacktestL1
    # max_position 1: two crossing buy events, the second cannot add
    bid = np.array([100., 100]); ask = np.array([100., 100])
    out = BacktestL1(max_position=1.0)(bid, ask, np.array([5., 5]), np.array([5., 5]),
                                       np.array([100., 100]), np.array([1., 1]),
                                       np.array([np.nan] * 2), np.array([np.nan] * 2))
    np.testing.assert_allclose(out[:, 2], [1, 1])        # capped at 1


def test_l1_stream_equals_batch():
    from screamer import BacktestL1
    rng = np.random.default_rng(2); n = 100
    mid = 100 + np.cumsum(rng.standard_normal(n) * 0.1)
    bid, ask = mid - 0.05, mid + 0.05
    out_op = BacktestL1(maker_fee=-0.0001)
    args = (bid, ask, np.full(n, 3.0), np.full(n, 3.0),
            bid, np.full(n, 1.0), ask, np.full(n, 1.0))
    stream = np.array([out_op(*(float(a[i]) for a in args)) for i in range(n)])
    batch = BacktestL1(maker_fee=-0.0001)(*args)
    np.testing.assert_allclose(np.nan_to_num(stream), np.nan_to_num(batch))
