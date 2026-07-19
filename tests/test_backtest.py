import math
import numpy as np
from screamer import BacktestSignal, backtest_report


def test_market_constant_and_encoding():
    from screamer import MARKET, BacktestSignal
    assert MARKET == math.inf
    # a directional market buy to +1 via MARKET fills fully at the price (frictionless)
    out = BacktestSignal()(np.array([1., 1]), np.array([100., 101.]))
    assert out[1, 2] == 1.0


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
    assert list(summary) == [
        "total_pnl", "max_drawdown", "total_cost", "turnover", "num_trades", "sharpe"]
    assert summary["num_trades"] == 2.0                    # enter long, exit
    assert summary["turnover"] == 2.0                      # 1 unit in, 1 out
    assert summary["max_drawdown"] <= 0.0
    assert summary["total_pnl"] == running["equity"][-1]
    assert summary["total_cost"] == float(np.nansum(out[:, 3]))
    # running is a plain dict of numpy arrays (no pandas), each column length T
    assert isinstance(running, dict) and isinstance(summary, dict)
    assert all(len(running[k]) == len(out) for k in
               ("equity", "drawdown", "cum_cost", "turnover", "trades", "sharpe"))


def test_backtest_report_node_matches_reference():
    from screamer import BacktestReport
    out = BacktestSignal()(np.sign(np.random.default_rng(0).standard_normal(200)),
                           100 + np.cumsum(np.random.default_rng(1).standard_normal(200)))
    eq, pnl, pos, cost = out[:, 0], out[:, 1], out[:, 2], out[:, 3]
    rep = BacktestReport()(eq, pnl, pos, cost)             # (T, 6)
    dd_ref = eq - np.maximum.accumulate(eq)                # dollar drawdown reference
    np.testing.assert_allclose(rep[:, 0], dd_ref)                          # drawdown
    np.testing.assert_allclose(rep[:, 1], np.cumsum(cost))                 # cum_cost
    np.testing.assert_allclose(rep[:, 4], np.minimum.accumulate(dd_ref))   # max_drawdown
    np.testing.assert_allclose(rep[:, 5][-1], pnl.mean() / pnl.std(ddof=1))  # sharpe


# --- BacktestOHLC ------------------------------------------------------------

def test_ohlc_target_is_deferred_one_bar_causal():
    from screamer import BacktestOHLC
    # target decided on bar t's close executes on bar t+1 (causal, no manual lag):
    # target=1 at bar 0 -> position becomes 1 at bar 1, then held.
    o = BacktestOHLC()(np.array([1., 1, 1, 0]), np.array([np.nan] * 4),
                       np.array([100., 101, 102, 103]), np.array([100., 101, 102, 103]),
                       np.array([100., 101, 102, 103]), np.array([100., 101, 102, 103]))
    np.testing.assert_allclose(o[:, 2], [0, 1, 1, 1])    # flat bar 0, long from bar 1 (deferred)
    np.testing.assert_allclose(o[:, 0], [0, 0, 1, 2])    # equity: long 1 earns 101->103 = +2


def test_ohlc_limit_fills_only_when_next_bar_reaches():
    from screamer import BacktestOHLC
    # a buy limit 99 decided on bar 0 rests during bar 1; it fills iff bar 1's low reaches it
    reached = BacktestOHLC()(np.array([1., 1.]), np.array([99., 99.]),
                             np.array([100., 100.]), np.array([101., 101.]),
                             np.array([100., 98.]), np.array([100., 100.]))
    assert reached[1, 2] == 1.0                           # bar 1 low 98 <= 99 -> fill
    missed = BacktestOHLC()(np.array([1., 1.]), np.array([99., 99.]),
                            np.array([100., 100.]), np.array([101., 101.]),
                            np.array([100., 99.5]), np.array([100., 100.]))
    assert missed[1, 2] == 0.0                            # bar 1 low 99.5 never reaches 99
    # touch vs breach at the exact level, on the executing bar (bar 1)
    touch = BacktestOHLC(fill="touch")(np.array([1., 1.]), np.array([99., 99.]),
                                       np.array([100., 100.]), np.array([101., 101.]),
                                       np.array([100., 99.]), np.array([100., 100.]))
    breach = BacktestOHLC(fill="breach")(np.array([1., 1.]), np.array([99., 99.]),
                                         np.array([100., 100.]), np.array([101., 101.]),
                                         np.array([100., 99.]), np.array([100., 100.]))
    assert touch[1, 2] == 1.0 and breach[1, 2] == 0.0


def test_ohlc_limit_fills_full_target_no_participation():
    from screamer import BacktestOHLC
    # a large target with a limit reached on the next bar fills fully (bars carry no volume)
    out = BacktestOHLC()(np.array([1000., 1000.]), np.array([99., 99.]),
                         np.array([100., 100.]), np.array([101., 101.]),
                         np.array([100., 98.]), np.array([100., 100.]))
    assert out[1, 2] == 1000.0


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

def test_l1_default_is_breach_full_fill_on_cross():
    from screamer import BacktestL1
    # default breach: my_bid 100 rests passive (ask 101), then ask crosses to 99 -> full fill at 100 (maker)
    out = BacktestL1()(np.array([100., 99.]), np.array([101., 99.5]),
                       np.array([5., 5]), np.array([5., 5]),
                       np.array([100., 100.]), np.array([10., 10.]),
                       np.array([np.nan, np.nan]), np.array([np.nan, np.nan]))
    assert out[0, 2] == 0.0                              # passive, no fill
    assert out[1, 2] == 10.0                             # swept: full 10 at my_bid 100
    assert out[1, 1] < 0.0                               # bought at 100, marks below -> adverse


def test_l1_breach_no_fill_on_lock():
    from screamer import BacktestL1
    # breach: ask == my_bid (locked) must NOT fill
    out = BacktestL1(fill="breach")(np.array([100.]), np.array([100.]),
                                    np.array([5.]), np.array([5.]),
                                    np.array([100.]), np.array([10.]),
                                    np.array([np.nan]), np.array([np.nan]))
    assert out[0, 2] == 0.0


def test_l1_touch_lock_fills_participation_once():
    from screamer import BacktestL1
    # touch: my_bid 100 resting; then ask locks at 100 (size 8), participation 0.5 -> min(10, 0.5*8)=4;
    # a second identical locked snapshot must NOT add (edge-triggered)
    bid = np.array([100., 100., 100.]); ask = np.array([101., 100., 100.])
    out = BacktestL1(fill="touch", participation_ratio=0.5)(
        bid, ask, np.array([5., 5, 5]), np.array([8., 8, 8]),
        np.array([100., 100, 100]), np.array([10., 10, 10]),
        np.array([np.nan] * 3), np.array([np.nan] * 3))
    assert out[0, 2] == 0.0                              # passive
    assert out[1, 2] == 4.0                              # lock entry: min(10, 0.5*8)
    assert out[2, 2] == 4.0                              # same lock: no further fill


def test_l1_submitted_crossing_is_taker_with_tick_slippage():
    from screamer import BacktestL1
    # quote appears already marketable (my_bid 100 > ask 99): taker, full fill, overflow at ask+tick.
    # ask_size 4, my_bid_size 10, tick 0.5 -> 4 @ 99, 6 @ 99.5; VWAP=(4*99+6*99.5)/10=99.3
    out = BacktestL1(taker_fee=0.0, tick_size=0.5)(
        np.array([98.]), np.array([99.]), np.array([5.]), np.array([4.]),
        np.array([100.]), np.array([10.]), np.array([np.nan]), np.array([np.nan]))
    assert out[0, 2] == 10.0                             # full taker fill
    # marks to mid (98+99)/2=98.5; bought VWAP 99.3 -> immediate adverse cost = 10*(99.3-98.5)
    np.testing.assert_allclose(out[0, 3], 10 * (99.3 - 98.5), atol=1e-9)


def test_l1_inventory_cap():
    from screamer import BacktestL1
    # ask 99 crosses through my_bid 100 (a taker sweep on first appearance);
    # room = max_position - 0 = 3 caps the fill at 3 of the 10 quoted
    out = BacktestL1(max_position=3.0)(
        np.array([98.]), np.array([99.]), np.array([5.]), np.array([5.]),
        np.array([100.]), np.array([10.]), np.array([np.nan]), np.array([np.nan]))
    assert out[0, 2] == 3.0                              # capped even on a full sweep


def test_l1_stream_equals_batch():
    from screamer import BacktestL1
    rng = np.random.default_rng(7); n = 200
    mid = 100 + np.cumsum(rng.standard_normal(n) * 0.1)
    bid, ask = mid - 0.05, mid + 0.05
    args = (bid, ask, np.full(n, 5.0), np.full(n, 5.0),
            bid - 0.01, np.full(n, 1.0), ask + 0.01, np.full(n, 1.0))
    op = BacktestL1()
    stream = np.array([op(*(float(a[i]) for a in args)) for i in range(n)])
    batch = BacktestL1()(*args)
    np.testing.assert_allclose(np.nan_to_num(stream), np.nan_to_num(batch))


# --- BacktestL1Trades --------------------------------------------------------

def test_l1trades_passive_fill_from_trade():
    from screamer import BacktestL1Trades
    # my_bid 100 resting inside (market 99.5/100.5); a sell-print AT 100 size 8, participation 0.5
    # -> min(10, 0.5*8)=4 at 100 (maker). NaN trade rows do not fill.
    out = BacktestL1Trades(participation_ratio=0.5)(
        np.array([99.5, 99.5]), np.array([100.5, 100.5]),
        np.array([5., 5]), np.array([5., 5]),
        np.array([100., 100.]), np.array([10., 10.]),
        np.array([np.nan, np.nan]), np.array([np.nan, np.nan]),
        np.array([np.nan, 100.]), np.array([np.nan, 8.]))     # quote row, then a trade row
    assert out[0, 2] == 0.0                                    # NaN trade -> no fill, mark only
    assert out[1, 2] == 4.0                                    # trade at 100 -> participation fill


def test_l1trades_through_trade_fills_full():
    from screamer import BacktestL1Trades
    # a sell-print THROUGH my_bid (99 < 100) fills the full resting size
    out = BacktestL1Trades()(
        np.array([99.5]), np.array([100.5]), np.array([5.]), np.array([5.]),
        np.array([100.]), np.array([10.]), np.array([np.nan]), np.array([np.nan]),
        np.array([99.]), np.array([2.]))
    assert out[0, 2] == 10.0


def test_l1trades_run_over_fills_without_a_trade():
    from screamer import BacktestL1Trades
    # no trade (NaN), but the quote ask crosses my_bid -> run-over full fill at my_bid
    out = BacktestL1Trades()(
        np.array([100., 99.]), np.array([101., 99.5]),
        np.array([5., 5]), np.array([5., 5]),
        np.array([100., 100.]), np.array([10., 10.]),
        np.array([np.nan, np.nan]), np.array([np.nan, np.nan]),
        np.array([np.nan, np.nan]), np.array([np.nan, np.nan]))
    assert out[0, 2] == 0.0                                    # passive
    assert out[1, 2] == 10.0                                   # run-over on the cross


def test_l1trades_submitted_crossing_is_taker():
    from screamer import BacktestL1Trades
    # my_bid 100 appears already marketable (ask 99): taker, full fill, overflow at ask+tick.
    # ask_size 4, size 10, tick 0.5 -> VWAP=(4*99+6*99.5)/10=99.3; mid=(98+99)/2=98.5
    out = BacktestL1Trades(tick_size=0.5)(
        np.array([98.]), np.array([99.]), np.array([5.]), np.array([4.]),
        np.array([100.]), np.array([10.]), np.array([np.nan]), np.array([np.nan]),
        np.array([np.nan]), np.array([np.nan]))
    assert out[0, 2] == 10.0
    np.testing.assert_allclose(out[0, 3], 10 * (99.3 - 98.5), atol=1e-9)


def test_l1trades_stream_equals_batch():
    from screamer import BacktestL1Trades
    rng = np.random.default_rng(9); n = 200
    mid = 100 + np.cumsum(rng.standard_normal(n) * 0.05)
    bid, ask = mid - 0.05, mid + 0.05
    tp = np.where(rng.standard_normal(n) > 0.5, mid, np.nan)   # sparse trades
    ts = np.where(np.isnan(tp), np.nan, 1.0)
    args = (bid, ask, np.full(n, 5.0), np.full(n, 5.0),
            bid, np.full(n, 1.0), ask, np.full(n, 1.0), tp, ts)
    op = BacktestL1Trades()
    stream = np.array([op(*(float(a[i]) for a in args)) for i in range(n)])
    batch = BacktestL1Trades()(*args)
    np.testing.assert_allclose(np.nan_to_num(stream), np.nan_to_num(batch))
