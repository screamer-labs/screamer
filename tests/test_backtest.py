import math
import numpy as np
from screamer import BacktestPriceTarget, backtest_report


def test_market_constant_and_encoding():
    from screamer import MARKET, BacktestPriceTarget
    assert MARKET == math.inf
    # a directional market buy to +1 via MARKET fills fully at the price (frictionless)
    out = BacktestPriceTarget()(np.array([1., 1]), np.array([100., 101.]))
    assert out[1, 2] == 1.0


def test_frictionless_marks_to_market():
    # hold long 1 through 100 -> 102, flat at the end
    out = BacktestPriceTarget()(np.array([1., 1, 0]), np.array([100., 101, 102]))
    np.testing.assert_allclose(out[:, 0], [0, 1, 2])       # equity
    np.testing.assert_allclose(out[:, 1], [0, 1, 1])       # pnl
    np.testing.assert_allclose(out[:, 2], [1, 1, 0])       # position
    np.testing.assert_allclose(out[:, 3], [0, 0, 0])       # cost


def test_frictionless_equals_cumsum_prev_position_times_dprice():
    rng = np.random.default_rng(0); n = 200
    price = 100 + np.cumsum(rng.standard_normal(n) * 0.3)
    signal = np.sign(rng.standard_normal(n))
    equity = BacktestPriceTarget()(signal, price)[:, 0]
    prev_pos = np.concatenate([[0.0], signal[:-1]])
    dprice = np.concatenate([[0.0], np.diff(price)])
    np.testing.assert_allclose(equity, np.cumsum(prev_pos * dprice), atol=1e-9)


def test_taker_cost_is_turnover_times_half_spread():
    # long then flat at a flat price: each trade crosses half of a 1% spread
    out = BacktestPriceTarget(spread=0.01)(np.array([1., 0]), np.array([100., 100]))
    np.testing.assert_allclose(out[:, 3], [0.5, 0.5])      # per-step cost
    np.testing.assert_allclose(out[-1, 0], -1.0)           # equity = -total cost


def test_fee_on_traded_notional():
    out = BacktestPriceTarget(fee=0.001)(np.array([2., 2]), np.array([100., 100]))
    np.testing.assert_allclose(out[:, 3], [0.2, 0.0])      # 2 units * 100 * 0.001, then no trade


def test_short_position_pnl():
    # short 1 while price rises 100 -> 102 loses 2
    out = BacktestPriceTarget()(np.array([-1., -1, 0]), np.array([100., 101, 102]))
    np.testing.assert_allclose(out[:, 0], [0, -1, -2])


def test_is_causal():
    price = np.array([100., 101, 102, 103]); signal = np.array([1., 1, -1, -1])
    full = BacktestPriceTarget()(signal, price)
    trunc = BacktestPriceTarget()(signal[:2], price[:2])
    np.testing.assert_allclose(full[:2], trunc)            # a future signal cannot change past rows


def test_stream_equals_batch():
    rng = np.random.default_rng(1); n = 200
    price = 100 + np.cumsum(rng.standard_normal(n) * 0.3)
    signal = np.sign(rng.standard_normal(n))
    op = BacktestPriceTarget(spread=0.0005, fee=0.0002)
    stream = np.array([op(float(s), float(p)) for s, p in zip(signal, price)])
    batch = BacktestPriceTarget(spread=0.0005, fee=0.0002)(signal, price)
    np.testing.assert_allclose(np.nan_to_num(stream), np.nan_to_num(batch))


def test_nan_skips_and_holds_position():
    out = BacktestPriceTarget()(np.array([1., np.nan, 1.]), np.array([100., 101, 102]))
    assert np.all(np.isnan(out[1]))                        # NaN bar -> all-NaN row, state untouched
    # position held across the gap; t2 marks 1*(102-100)=2 against the last good price
    np.testing.assert_allclose(out[2], [2, 2, 1, 0])


def test_reset_restarts():
    price = np.array([100., 101, 100]); signal = np.array([1., 1, 1])
    op = BacktestPriceTarget()
    a = op(signal, price); op.reset(); b = op(signal, price)
    np.testing.assert_allclose(a, b)


def test_backtest_report_shape_and_invariants():
    out = BacktestPriceTarget(spread=0.01)(np.array([1., 1, 0]), np.array([100., 90, 95]))
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
    out = BacktestPriceTarget()(np.sign(np.random.default_rng(0).standard_normal(200)),
                           100 + np.cumsum(np.random.default_rng(1).standard_normal(200)))
    eq, pnl, pos, cost = out[:, 0], out[:, 1], out[:, 2], out[:, 3]
    rep = BacktestReport()(eq, pnl, pos, cost)             # (T, 6)
    dd_ref = eq - np.maximum.accumulate(eq)                # dollar drawdown reference
    np.testing.assert_allclose(rep[:, 0], dd_ref)                          # drawdown
    np.testing.assert_allclose(rep[:, 1], np.cumsum(cost))                 # cum_cost
    np.testing.assert_allclose(rep[:, 4], np.minimum.accumulate(dd_ref))   # max_drawdown
    np.testing.assert_allclose(rep[:, 5][-1], pnl.mean() / pnl.std(ddof=1))  # sharpe


def test_signal_position_cap_clamps_target():
    from screamer import BacktestPriceTarget
    import numpy as np
    out = BacktestPriceTarget(max_position=1.0, min_position=-1.0)(
        np.array([5., -5., 0.]), np.array([100., 100., 100.]))
    np.testing.assert_allclose(out[:, 2], [1, -1, 0])   # target 5 clamped to 1, -5 to -1


def test_price_target_reaches_target_and_costs():
    import numpy as np
    from screamer import BacktestPriceTarget
    out = BacktestPriceTarget(spread=0.0, fee=0.001)(
        np.array([1., 1., 0.]), np.array([100., 110., 121.]))
    np.testing.assert_allclose(out[:, 2], [1., 1., 0.])          # position track
    assert out[0, 3] > 0.0                                        # taker fee charged on the buy


# --- BacktestOHLCTarget (causal market-at-open) --------------------------------

def test_ohlc_target_is_deferred_one_bar_causal():
    from screamer import BacktestOHLCTarget
    # target decided on bar t's close executes on bar t+1 (causal, no manual lag):
    # target=1 at bar 0 -> position becomes 1 at bar 1, then held.
    o = BacktestOHLCTarget()(np.array([1., 1, 1, 0]),
                              np.array([100., 101, 102, 103]), np.array([100., 101, 102, 103]),
                              np.array([100., 101, 102, 103]), np.array([100., 101, 102, 103]))
    np.testing.assert_allclose(o[:, 2], [0, 1, 1, 1])    # flat bar 0, long from bar 1 (deferred)
    np.testing.assert_allclose(o[:, 0], [0, 0, 1, 2])    # equity: long 1 earns 101->103 = +2


def test_ohlc_target_stream_equals_batch_and_reset():
    from screamer import BacktestOHLCTarget
    rng = np.random.default_rng(0); n = 100
    c = 100 + np.cumsum(rng.standard_normal(n) * 0.3)
    o, h, l = c - 0.1, c + 0.5, c - 0.5
    tgt = np.sign(rng.standard_normal(n))
    op = BacktestOHLCTarget(taker_fee=0.001)
    stream = np.array([op(float(tgt[i]), float(o[i]), float(h[i]), float(l[i]), float(c[i]))
                       for i in range(n)])
    op.reset()
    batch = BacktestOHLCTarget(taker_fee=0.001)(tgt, o, h, l, c)
    np.testing.assert_allclose(np.nan_to_num(stream), np.nan_to_num(batch))


def test_ohlc_target_position_cap_clamps_target():
    from screamer import BacktestOHLCTarget
    import numpy as np
    # target 5 decided on bar 0, executed on bar 1 (deferred), clamped to max 1
    out = BacktestOHLCTarget(max_position=1.0)(
        np.array([5., 5.]),
        np.array([100., 100.]), np.array([101., 101.]),
        np.array([100., 100.]), np.array([100., 100.]))
    assert out[1, 2] == 1.0


# --- BacktestTrades (now BacktestTradesOrders one-sided) ----------------------
# BacktestTrades accepted signed order_size (positive=buy, negative=sell). The
# same behaviour is available from BacktestTradesOrders by zeroing the idle side.

def test_trades_fill_and_adverse_selection():
    from screamer import BacktestTradesOrders
    # resting buy 1 @ 100; a print at 99 (<=100) size 2 fills 1 @ 100, marks at 99 (down 1);
    # then a print at 101 with the order still resting (101 > 100, no fill) marks the position up
    t = BacktestTradesOrders()(np.array([100., 100]), np.array([1., 1]),
                               np.array([np.nan, np.nan]), np.array([0., 0.]),
                               np.array([99., 101]), np.array([2., 5]))
    np.testing.assert_allclose(t[:, 2], [1, 1])          # position filled then held
    np.testing.assert_allclose(t[0], [-1, -1, 1, 1])     # bought at 100 vs 99 mark -> cost 1
    np.testing.assert_allclose(t[1, 0], 1.0)             # mark 99 -> 101 recovers to +1


def test_trades_partial_up_to_print_size():
    from screamer import BacktestTradesOrders
    # resting buy 5 @ 100; a print AT 100 size 2 fills min(5, 1.0*2)=2 (participation 1.0)
    t = BacktestTradesOrders()(np.array([100.]), np.array([5.]),
                               np.array([np.nan]), np.array([0.]),
                               np.array([100.]), np.array([2.]))
    assert t[0, 2] == 2.0


def test_trades_through_fills_full_order_not_print_size():
    from screamer import BacktestTradesOrders
    # resting buy 10 @ 100; a print of size 2 at 99 trades THROUGH -> full 10 fills
    t = BacktestTradesOrders()(np.array([100.]), np.array([10.]),
                               np.array([np.nan]), np.array([0.]),
                               np.array([99.]), np.array([2.]))
    assert t[0, 2] == 10.0                                  # swept: full order, not min(10,2)


def test_trades_at_fills_participation_of_trade_size():
    from screamer import BacktestTradesOrders
    # resting buy 10 @ 100; a print of size 8 AT 100, participation 0.5 -> min(10, 0.5*8)=4
    t = BacktestTradesOrders(participation_ratio=0.5)(
        np.array([100.]), np.array([10.]),
        np.array([np.nan]), np.array([0.]),
        np.array([100.]), np.array([8.]))
    assert t[0, 2] == 4.0


def test_trades_participation_capped_by_order_no_zeno():
    from screamer import BacktestTradesOrders
    # participation*trade_size exceeds the order -> full fill, capped by remaining
    t = BacktestTradesOrders(participation_ratio=0.5)(
        np.array([100.]), np.array([10.]),
        np.array([np.nan]), np.array([0.]),
        np.array([100.]), np.array([100.]))
    assert t[0, 2] == 10.0                                  # min(10, 0.5*100)=10, no dust


def test_trades_breach_ignores_at_price():
    from screamer import BacktestTradesOrders
    # breach: a print exactly AT 100 does not fill; only strictly through does
    at = BacktestTradesOrders(fill="breach")(
        np.array([100.]), np.array([10.]),
        np.array([np.nan]), np.array([0.]),
        np.array([100.]), np.array([8.]))
    assert at[0, 2] == 0.0
    through = BacktestTradesOrders(fill="breach")(
        np.array([100.]), np.array([10.]),
        np.array([np.nan]), np.array([0.]),
        np.array([99.]), np.array([2.]))
    assert through[0, 2] == 10.0


def test_trades_stream_equals_batch():
    from screamer import BacktestTradesOrders
    rng = np.random.default_rng(1); n = 100
    price = 100 + np.cumsum(rng.standard_normal(n) * 0.2)
    op = np.where(rng.standard_normal(n) > 0, price - 0.1, price + 0.1)   # order near the print
    os_ = rng.choice([-1.0, 1.0], n)                                       # signed: +1=buy, -1=sell
    ts = np.abs(rng.standard_normal(n)) + 1
    # split signed order into explicit bid/ask sides
    bid_p = np.where(os_ > 0, op, np.nan)
    bid_s = np.where(os_ > 0, 1.0, 0.0)
    ask_p = np.where(os_ < 0, op, np.nan)
    ask_s = np.where(os_ < 0, 1.0, 0.0)
    eng = BacktestTradesOrders(maker_fee=-0.0001)
    stream = np.array([eng(bid_p[i], bid_s[i], ask_p[i], ask_s[i], price[i], ts[i])
                       for i in range(n)])
    eng.reset()
    batch = BacktestTradesOrders(maker_fee=-0.0001)(bid_p, bid_s, ask_p, ask_s, price, ts)
    np.testing.assert_allclose(np.nan_to_num(stream), np.nan_to_num(batch))


def test_trades_fill_truncated_by_cap():
    from screamer import BacktestTradesOrders
    import numpy as np
    # resting buy 10 @ 100, a through-print would fill 10, but max_position 3 caps it
    out = BacktestTradesOrders(max_position=3.0)(
        np.array([100.]), np.array([10.]),
        np.array([np.nan]), np.array([0.]),
        np.array([99.]), np.array([2.]))
    assert out[0, 2] == 3.0


# --- BacktestL1 (now BacktestL1Orders) ----------------------------------------

def test_l1_default_is_breach_full_fill_on_cross():
    from screamer import BacktestL1Orders
    # default breach: my_bid 100 rests passive (market_ask 101), then market_ask crosses to 99
    # -> full fill at 100 (maker). Input order: (my_bid, my_bid_size, my_ask, my_ask_size,
    # market_bid, market_ask, market_bid_size, market_ask_size)
    out = BacktestL1Orders()(
        np.array([100., 100.]), np.array([10., 10.]),
        np.array([np.nan, np.nan]), np.array([np.nan, np.nan]),
        np.array([100., 99.]), np.array([101., 99.5]),
        np.array([5., 5]), np.array([5., 5]))
    assert out[0, 2] == 0.0                              # passive, no fill
    assert out[1, 2] == 10.0                             # swept: full 10 at my_bid 100
    assert out[1, 1] < 0.0                               # bought at 100, marks below -> adverse


def test_l1_breach_no_fill_on_lock():
    from screamer import BacktestL1Orders
    # breach: market_ask == my_bid (locked) must NOT fill
    out = BacktestL1Orders(fill="breach")(
        np.array([100.]), np.array([10.]),
        np.array([np.nan]), np.array([np.nan]),
        np.array([100.]), np.array([100.]),
        np.array([5.]), np.array([5.]))
    assert out[0, 2] == 0.0


def test_l1_touch_lock_fills_participation_once():
    from screamer import BacktestL1Orders
    # touch: my_bid 100 resting; then market_ask locks at 100 (size 8),
    # participation 0.5 -> min(10, 0.5*8)=4;
    # a second identical locked snapshot must NOT add (edge-triggered)
    out = BacktestL1Orders(fill="touch", participation_ratio=0.5)(
        np.array([100., 100., 100.]), np.array([10., 10, 10]),
        np.array([np.nan] * 3), np.array([np.nan] * 3),
        np.array([100., 100., 100.]), np.array([101., 100., 100.]),
        np.array([5., 5, 5]), np.array([8., 8, 8]))
    assert out[0, 2] == 0.0                              # passive
    assert out[1, 2] == 4.0                              # lock entry: min(10, 0.5*8)
    assert out[2, 2] == 4.0                              # same lock: no further fill


def test_l1_submitted_crossing_is_taker_with_tick_slippage():
    from screamer import BacktestL1Orders
    # quote appears already marketable (my_bid 100 > market_ask 99): taker, full fill, overflow at
    # market_ask+tick. market_ask_size 4, my_bid_size 10, tick 0.5 -> 4 @ 99, 6 @ 99.5;
    # VWAP=(4*99+6*99.5)/10=99.3
    out = BacktestL1Orders(taker_fee=0.0, tick_size=0.5)(
        np.array([100.]), np.array([10.]),
        np.array([np.nan]), np.array([np.nan]),
        np.array([98.]), np.array([99.]),
        np.array([5.]), np.array([4.]))
    assert out[0, 2] == 10.0                             # full taker fill
    # marks to mid (98+99)/2=98.5; bought VWAP 99.3 -> immediate adverse cost = 10*(99.3-98.5)
    np.testing.assert_allclose(out[0, 3], 10 * (99.3 - 98.5), atol=1e-9)


def test_l1_inventory_cap():
    from screamer import BacktestL1Orders
    # market_ask 99 crosses through my_bid 100 (a taker sweep on first appearance);
    # room = max_position - 0 = 3 caps the fill at 3 of the 10 quoted
    out = BacktestL1Orders(max_position=3.0)(
        np.array([100.]), np.array([10.]),
        np.array([np.nan]), np.array([np.nan]),
        np.array([98.]), np.array([99.]),
        np.array([5.]), np.array([5.]))
    assert out[0, 2] == 3.0                              # capped even on a full sweep


def test_l1_stream_equals_batch():
    from screamer import BacktestL1Orders
    rng = np.random.default_rng(7); n = 200
    mid = 100 + np.cumsum(rng.standard_normal(n) * 0.1)
    market_bid, market_ask = mid - 0.05, mid + 0.05
    my_bid, my_ask = market_bid - 0.01, market_ask + 0.01
    args = (my_bid, np.full(n, 1.0), my_ask, np.full(n, 1.0),
            market_bid, market_ask, np.full(n, 5.0), np.full(n, 5.0))
    op = BacktestL1Orders()
    stream = np.array([op(*(float(a[i]) for a in args)) for i in range(n)])
    batch = BacktestL1Orders()(*args)
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


# --- BacktestOHLCOrders ------------------------------------------------------

def test_ohlc_orders_two_sided_fills_on_range():
    from screamer import BacktestOHLCOrders
    import numpy as np
    # bid 99 rests; the bar low 98 reaches it -> buy 1 at 99; no ask this bar
    out = BacktestOHLCOrders()(
        np.array([99.]), np.array([1.]), np.array([np.nan]), np.array([0.]),
        np.array([100.]), np.array([101.]), np.array([98.]), np.array([100.]))
    assert out[0, 2] == 1.0                       # bought 1 at the bid
    # marks to close 100 vs fill 99 -> +1 mark, cost 0 (maker), equity +1
    np.testing.assert_allclose(out[0, 0], 1.0, atol=1e-9)

def test_ohlc_orders_inventory_cap():
    from screamer import BacktestOHLCOrders
    import numpy as np
    # bid 99 size 10, low reaches it, but max_position 2 caps the buy
    out = BacktestOHLCOrders(max_position=2.0)(
        np.array([99.]), np.array([10.]), np.array([np.nan]), np.array([0.]),
        np.array([100.]), np.array([101.]), np.array([98.]), np.array([100.]))
    assert out[0, 2] == 2.0

def test_ohlc_orders_stream_equals_batch():
    from screamer import BacktestOHLCOrders
    import numpy as np
    rng = np.random.default_rng(0); n = 200
    close = 100 + np.cumsum(rng.standard_normal(n) * 0.2)
    o, h, l = close - 0.1, close + 0.3, close - 0.3
    bid, ask = close - 0.2, close + 0.2
    one = np.ones(n)
    args = (bid, one, ask, one, o, h, l, close)
    op = BacktestOHLCOrders(max_position=5.0, min_position=-5.0)
    stream = np.array([op(*(float(a[i]) for a in args)) for i in range(n)])
    op.reset()
    batch = BacktestOHLCOrders(max_position=5.0, min_position=-5.0)(*args)
    np.testing.assert_allclose(np.nan_to_num(stream), np.nan_to_num(batch))

def test_ohlc_orders_market_buy_fills_at_open():
    from screamer import BacktestOHLCOrders, MARKET
    import numpy as np
    # a market bid (inf price via MARKET) fills at the open as a taker, not at the low
    out = BacktestOHLCOrders()(
        np.array([MARKET]), np.array([1.]), np.array([np.nan]), np.array([0.]),
        np.array([100.]), np.array([101.]), np.array([99.5]), np.array([100.]))
    assert out[0, 2] == 1.0        # bought 1 at the open (market order)


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


# --- BacktestTradesOrders (two-sided fills on prints) ------------------------

def test_trades_maker_two_sided_fills_on_prints():
    from screamer import BacktestTradesOrders
    import numpy as np
    # resting bid 100 size 5; a sell-print at 99 (<=100) size 8, participation 1.0 -> buy 5 at 100
    out = BacktestTradesOrders()(
        np.array([100.]), np.array([5.]), np.array([np.nan]), np.array([0.]),
        np.array([99.]), np.array([8.]))
    assert out[0, 2] == 5.0

def test_trades_maker_cap_and_participation():
    from screamer import BacktestTradesOrders
    import numpy as np
    # at-price print size 8, participation 0.5 -> min(remaining, 0.5*8)=4, capped by max 3
    out = BacktestTradesOrders(participation_ratio=0.5, max_position=3.0)(
        np.array([100.]), np.array([10.]), np.array([np.nan]), np.array([0.]),
        np.array([100.]), np.array([8.]))
    assert out[0, 2] == 3.0

def test_trades_maker_stream_equals_batch():
    from screamer import BacktestTradesOrders
    import numpy as np
    rng = np.random.default_rng(1); n = 200
    price = 100 + np.cumsum(rng.standard_normal(n) * 0.1)
    size = np.abs(rng.standard_normal(n)) + 0.5
    bid, ask = price + 0.05, price - 0.05
    one = np.ones(n)
    args = (bid, one, ask, one, price, size)
    op = BacktestTradesOrders(max_position=8.0, min_position=-8.0)
    stream = np.array([op(*(float(a[i]) for a in args)) for i in range(n)])
    op.reset()
    batch = BacktestTradesOrders(max_position=8.0, min_position=-8.0)(*args)
    np.testing.assert_allclose(np.nan_to_num(stream), np.nan_to_num(batch))

def test_trades_maker_market_buy_fills_on_any_print():
    from screamer import BacktestTradesOrders, MARKET
    import numpy as np
    # a market bid (inf price via MARKET) is swept by any print as a taker
    out = BacktestTradesOrders()(
        np.array([MARKET]), np.array([2.]), np.array([np.nan]), np.array([0.]),
        np.array([101.]), np.array([5.]))
    assert out[0, 2] == 2.0        # bought 2 (bid_size) against the print


# --- BacktestOHLCOrders ------------------------------------------------------

def test_ohlc_orders_two_sided_fill():
    import numpy as np
    from screamer import BacktestOHLCOrders
    out = BacktestOHLCOrders()(
        np.array([99.]), np.array([1.]), np.array([np.nan]), np.array([0.]),
        np.array([100.]), np.array([101.]), np.array([98.]), np.array([100.]))
    assert out[0, 2] == 1.0                                # resting bid hit on the low


# --- BacktestOHLCTarget ------------------------------------------------------

def test_ohlc_target_defers_to_next_open():
    import numpy as np
    from screamer import BacktestOHLCTarget
    # target +1 decided on bar 0 executes at bar 1's open (deferred); bar 0 stays flat
    out = BacktestOHLCTarget()(
        np.array([1., 1.]), np.array([100., 105.]),
        np.array([100., 105.]), np.array([100., 105.]), np.array([100., 105.]))
    assert out[0, 2] == 0.0                                # nothing executes on bar 0
    assert out[1, 2] == 1.0                                # filled at bar 1 open

def test_ohlc_target_market_capped():
    import numpy as np
    from screamer import BacktestOHLCTarget
    out = BacktestOHLCTarget(max_position=1.0)(
        np.array([5., 5.]), np.array([100., 100.]),
        np.array([100., 100.]), np.array([100., 100.]), np.array([100., 100.]))
    assert out[1, 2] == 1.0                                # target 5 clamped to the cap


# --- BacktestTradesOrders / BacktestTradesTarget ---------------------------------

def test_trades_orders_one_sided_equals_resting_limit():
    import numpy as np
    from screamer import BacktestTradesOrders
    # a resting bid at 100 size 5; a sell-print at 99 sweeps it -> buy 5
    out = BacktestTradesOrders()(
        np.array([100.]), np.array([5.]), np.array([np.nan]), np.array([0.]),
        np.array([99.]), np.array([8.]))
    assert out[0, 2] == 5.0

def test_trades_target_takes_prints_to_reach_target():
    import numpy as np
    from screamer import BacktestTradesTarget
    # target +2 taken against the print; participation caps to the print size where needed
    out = BacktestTradesTarget()(
        np.array([2., 2.]), np.array([100., 100.]), np.array([10., 10.]))
    assert out[-1, 2] == 2.0

def test_trades_target_capped():
    import numpy as np
    from screamer import BacktestTradesTarget
    out = BacktestTradesTarget(max_position=1.0)(
        np.array([9.]), np.array([100.]), np.array([100.]))
    assert out[0, 2] == 1.0


# --- BacktestL1Orders ---------------------------------------------------------

def test_l1_orders_parity_resting_quote():
    import numpy as np
    from screamer import BacktestL1Orders
    # resting bid at 100 fills when the market ask drops to it (breach default)
    out = BacktestL1Orders()(
        np.array([100.]), np.array([1.]), np.array([np.nan]), np.array([0.]),
        np.array([100.5]), np.array([99.9]), np.array([5.]), np.array([5.]))
    assert out[0, 2] == 1.0


# --- BacktestL1Target ---------------------------------------------------------

def test_l1_target_takes_book_to_reach_target():
    import numpy as np
    from screamer import BacktestL1Target
    # target +1 taken against the displayed ask
    out = BacktestL1Target()(
        np.array([1.]), np.array([100.]), np.array([100.1]),
        np.array([5.]), np.array([5.]))
    assert out[0, 2] == 1.0


def test_l1_target_capped():
    import numpy as np
    from screamer import BacktestL1Target
    out = BacktestL1Target(max_position=1.0)(
        np.array([9.]), np.array([100.]), np.array([100.1]),
        np.array([50.]), np.array([50.]))
    assert out[0, 2] == 1.0
