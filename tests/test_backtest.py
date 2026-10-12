import pytest
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
        "total_pnl", "max_drawdown", "total_cost", "turnover", "num_trades", "sharpe",
        "exposure", "round_trips", "win_rate", "profit_factor", "avg_win", "avg_loss",
        "avg_holding", "sortino", "calmar"]
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

def test_contract_multiplier_and_fixed_fee_are_generic():
    from screamer import BacktestPriceTarget
    out = BacktestPriceTarget(multiplier=50.0, fee_per_contract=2.0)(
        np.array([1., 1., 0.]), np.array([100., 101., 101.]))
    # The fixed fee is per unit; the price move is 50 currency units per contract.
    np.testing.assert_allclose(out[:, 3], [2.0, 0.0, 2.0])
    np.testing.assert_allclose(out[:, 0], [-2.0, 48.0, 46.0])


def test_ohlc_target_uses_contract_multiplier_and_fixed_taker_fee():
    from screamer import BacktestOHLCTarget
    out = BacktestOHLCTarget(multiplier=10.0, fee_per_contract=1.5)(
        np.array([1., 1.]), np.array([100., 101.]),
        np.array([100., 101.]), np.array([100., 101.]),
        np.array([100., 101.]))
    # Target at bar 0 executes at bar 1 open: fixed fee is visible on that fill.
    np.testing.assert_allclose(out[:, 2], [0.0, 1.0])
    np.testing.assert_allclose(out[1, 3], 1.5)


def test_portfolio_report_reduces_assets_without_hiding_offsetting_turnover():
    from screamer import PortfolioReport
    engine = np.array([
        [[0.0, 0.0, 1.0, 1.0], [0.0, 0.0, -1.0, 2.0]],
        [[10.0, 10.0, 1.0, 0.0], [5.0, 5.0, 0.0, 1.0]],
    ])
    batch = PortfolioReport()(engine)
    np.testing.assert_allclose(batch[:, :5], [
        [0.0, 3.0, 2.0, 2.0, 0.0],
        [0.0, 4.0, 3.0, 3.0, 0.0],
    ])
    np.testing.assert_allclose(batch[1, 5], np.mean([0.0, 15.0]) / np.std([0.0, 15.0], ddof=1))

    op = PortfolioReport()
    streamed = np.array([op(row) for row in engine])
    np.testing.assert_allclose(np.nan_to_num(streamed), np.nan_to_num(batch))

    lazy = np.array(list(PortfolioReport()(iter(engine))))
    np.testing.assert_allclose(np.nan_to_num(lazy), np.nan_to_num(batch))


def test_portfolio_report_nan_row_holds_state():
    from screamer import PortfolioReport
    rows = np.array([
        [[0.0, 0.0, 1.0, 0.0]],
        [[np.nan, 1.0, 1.0, 0.0]],
        [[2.0, 2.0, 1.0, 0.0]],
    ])
    out = PortfolioReport()(rows)
    assert np.all(np.isnan(out[1]))
    np.testing.assert_allclose(out[2, :4], [0.0, 0.0, 1.0, 1.0])


def test_portfolio_report_rejects_asset_count_change():
    from screamer import PortfolioReport
    op = PortfolioReport()
    op(np.zeros((2, 4)))
    with pytest.raises(ValueError):
        op(np.zeros((3, 4)))


def _priced_book():
    T, K = 3, 2
    out = np.zeros((T, K, 4))
    out[:, :, 2] = [[0, 0], [1, 2], [1, 0]]          # positions in units
    prices = np.array([[10., 1.], [10., 1.], [20., 1.]])
    return out, prices


def test_portfolio_report_turnover_with_prices_is_traded_notional():
    from screamer import PortfolioReport
    out, prices = _priced_book()
    rep = PortfolioReport()(out, prices)
    # step 1: |1|*10 + |2|*1 = 12 ; step 2: 0*20 + |-2|*1 = 2 ; cumulative
    np.testing.assert_array_equal(rep[:, 2], [0., 12., 14.])
    np.testing.assert_array_equal(rep[:, 3], [0., 2., 3.])          # trades unchanged
    np.testing.assert_array_equal(PortfolioReport()(out)[:, 2], [0., 3., 5.])


def test_portfolio_report_prices_stream_equals_batch():
    import asyncio
    from screamer import PortfolioReport
    out, prices = _priced_book()
    batch = PortfolioReport()(out, prices)
    two = np.array(list(PortfolioReport()(iter(out), iter(prices))))
    pairs = np.array(list(PortfolioReport()(zip(out, prices))))
    op = PortfolioReport()
    live = np.array([op(row, p) for row, p in zip(out, prices)])

    async def agen(x):
        for v in x:
            yield v

    async def run():
        return [r async for r in PortfolioReport()(agen(out), agen(prices))]

    aio = np.array(asyncio.run(run()))
    for got in (two, pairs, live, aio):
        np.testing.assert_array_equal(got, batch)


def test_portfolio_report_prices_containers():
    from screamer import PortfolioReport
    out, prices = _priced_book()
    batch = PortfolioReport()(out, prices)
    as_list = PortfolioReport()(out.tolist(), prices.tolist())
    assert isinstance(as_list, list)
    np.testing.assert_array_equal(np.array(as_list), batch)
    for mixed in ((out.tolist(), prices), (out, prices.tolist())):
        got = PortfolioReport()(*mixed)
        assert isinstance(got, np.ndarray)                          # A8
        np.testing.assert_array_equal(got, batch)


def test_portfolio_report_prices_of_one_match_no_prices():
    from screamer import PortfolioReport
    out = np.random.default_rng(0).random((20, 3, 4))
    np.testing.assert_array_equal(PortfolioReport()(out, np.ones((20, 3))),
                                  PortfolioReport()(out))


def test_portfolio_report_nan_price_skips_row():
    from screamer import PortfolioReport
    out, prices = _priced_book()
    prices[1, 0] = np.nan
    rep = PortfolioReport()(out, prices)
    assert np.all(np.isnan(rep[1]))
    # the row is skipped, so step 2 trades the full change from step 0
    np.testing.assert_array_equal(rep[2, 2], 20.0)


def test_portfolio_report_nan_price_of_an_untraded_asset_is_not_read():
    from screamer import PortfolioReport
    out, prices = _priced_book()
    prices[2, 0] = np.nan                     # asset 0 holds 1 unit at step 2
    prices[0, :] = np.nan                     # nothing trades at step 0
    np.testing.assert_array_equal(PortfolioReport()(out, prices)[:, 2], [0., 12., 14.])


def test_portfolio_report_prices_argument_errors():
    from screamer import PortfolioReport
    out, prices = _priced_book()
    with pytest.raises(ValueError):
        PortfolioReport()(out, prices[:2])                           # T mismatch
    with pytest.raises(ValueError):
        PortfolioReport()(out, prices[:, :1])                        # K mismatch
    with pytest.raises(ValueError):
        PortfolioReport()(out[1], prices[1, :1])
    with pytest.raises(TypeError):
        PortfolioReport()(out, "prices")
    with pytest.raises(TypeError):
        PortfolioReport()(iter(out), prices)


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


# --- BacktestL1TradesOrders (renamed from BacktestL1Trades) ------------------

def test_l1trades_orders_exists_and_fills():
    import numpy as np
    from screamer import BacktestL1TradesOrders
    # resting bid filled when a trade explains the cross
    out = BacktestL1TradesOrders()(
        np.array([100.]), np.array([1.]), np.array([np.nan]), np.array([0.]),
        np.array([100.5]), np.array([99.9]), np.array([5.]), np.array([5.]),
        np.array([99.9]), np.array([3.]))
    assert out[0, 2] == 1.0


def test_l1trades_passive_fill_from_trade():
    from screamer import BacktestL1TradesOrders
    # my_bid 100 resting inside (market 99.5/100.5); a sell-print AT 100 size 8, participation 0.5
    # -> min(10, 0.5*8)=4 at 100 (maker). NaN trade rows do not fill.
    out = BacktestL1TradesOrders(participation_ratio=0.5)(
        np.array([100., 100.]), np.array([10., 10.]),
        np.array([np.nan, np.nan]), np.array([np.nan, np.nan]),
        np.array([99.5, 99.5]), np.array([100.5, 100.5]),
        np.array([5., 5]), np.array([5., 5]),
        np.array([np.nan, 100.]), np.array([np.nan, 8.]))     # quote row, then a trade row
    assert out[0, 2] == 0.0                                    # NaN trade -> no fill, mark only
    assert out[1, 2] == 4.0                                    # trade at 100 -> participation fill


def test_l1trades_through_trade_fills_full():
    from screamer import BacktestL1TradesOrders
    # a sell-print THROUGH my_bid (99 < 100) fills the full resting size
    out = BacktestL1TradesOrders()(
        np.array([100.]), np.array([10.]), np.array([np.nan]), np.array([np.nan]),
        np.array([99.5]), np.array([100.5]), np.array([5.]), np.array([5.]),
        np.array([99.]), np.array([2.]))
    assert out[0, 2] == 10.0


def test_l1trades_run_over_fills_without_a_trade():
    from screamer import BacktestL1TradesOrders
    # no trade (NaN), but the quote ask crosses my_bid -> run-over full fill at my_bid
    out = BacktestL1TradesOrders()(
        np.array([100., 100.]), np.array([10., 10.]),
        np.array([np.nan, np.nan]), np.array([np.nan, np.nan]),
        np.array([100., 99.]), np.array([101., 99.5]),
        np.array([5., 5]), np.array([5., 5]),
        np.array([np.nan, np.nan]), np.array([np.nan, np.nan]))
    assert out[0, 2] == 0.0                                    # passive
    assert out[1, 2] == 10.0                                   # run-over on the cross


def test_l1trades_submitted_crossing_is_taker():
    from screamer import BacktestL1TradesOrders
    # my_bid 100 appears already marketable (ask 99): taker, full fill, overflow at ask+tick.
    # ask_size 4, size 10, tick 0.5 -> VWAP=(4*99+6*99.5)/10=99.3; mid=(98+99)/2=98.5
    out = BacktestL1TradesOrders(tick_size=0.5)(
        np.array([100.]), np.array([10.]), np.array([np.nan]), np.array([np.nan]),
        np.array([98.]), np.array([99.]), np.array([5.]), np.array([4.]),
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
    from screamer import BacktestL1TradesOrders
    rng = np.random.default_rng(9); n = 200
    mid = 100 + np.cumsum(rng.standard_normal(n) * 0.05)
    bid, ask = mid - 0.05, mid + 0.05
    tp = np.where(rng.standard_normal(n) > 0.5, mid, np.nan)   # sparse trades
    ts = np.where(np.isnan(tp), np.nan, 1.0)
    args = (bid, np.full(n, 1.0), ask, np.full(n, 1.0),        # my_bid, my_bid_size, my_ask, my_ask_size
            bid, ask, np.full(n, 5.0), np.full(n, 5.0), tp, ts) # market_bid, market_ask, market_bid_size, market_ask_size, trade_price, trade_size
    op = BacktestL1TradesOrders()
    stream = np.array([op(*(float(a[i]) for a in args)) for i in range(n)])
    batch = BacktestL1TradesOrders()(*args)
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
    # bid 100 is submitted already crossing market_ask 99.9: taker sweep, not a resting maker fill
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


# --- Cost reference, slippage units, fill_at and delay ------------------------

def _bars():
    # bar 0: decision bar; bar 1: execution bar (open 101, close 101.5)
    o = np.array([100., 101.]); h = np.array([100.5, 102.])
    l = np.array([99.5, 100.6]); c = np.array([100., 101.5])
    return o, h, l, c


def test_ohlc_target_cost_is_friction_not_intrabar_move():
    from screamer import BacktestOHLCTarget
    o, h, l, c = _bars()
    out = BacktestOHLCTarget()(np.array([1., 1.]), o, h, l, c)
    # frictionless fill at the 101 open: the open -> close move is pnl, not cost
    np.testing.assert_allclose(out[1], [0.5, 0.5, 1.0, 0.0])


def test_ohlc_target_cost_equals_slippage_plus_fee():
    from screamer import BacktestOHLCTarget
    o, h, l, c = _bars()
    out = BacktestOHLCTarget(taker_fee=1e-3, slippage=0.05, slippage_unit="price")(
        np.array([1., 1.]), o, h, l, c)
    fill = 101.05
    np.testing.assert_allclose(out[1, 3], 0.05 + fill * 1e-3)        # slippage + fee
    np.testing.assert_allclose(out[1, 0], 101.5 - fill - fill * 1e-3)  # equity


def test_cost_reference_never_changes_equity():
    # The engine's equity must equal a hand-built fill-to-mark account.
    from screamer import BacktestOHLCTarget
    rng = np.random.default_rng(1); n = 300
    c = 100 + np.cumsum(rng.standard_normal(n) * 0.3)
    o = np.r_[c[0], c[:-1]] + rng.standard_normal(n) * 0.1
    h = np.maximum(o, c) + 0.2; l = np.minimum(o, c) - 0.2
    tgt = np.sign(rng.standard_normal(n))
    out = BacktestOHLCTarget(taker_fee=5e-4, slippage=2e-4)(tgt, o, h, l, c)
    pos = np.r_[0.0, tgt[:-1]]
    dpos = np.diff(np.r_[0.0, pos])
    fill = o * (1 + np.sign(dpos) * 2e-4)
    pnl = np.r_[0.0, pos[:-1] * np.diff(c)] + dpos * (c - fill) - np.abs(dpos) * fill * 5e-4
    np.testing.assert_allclose(out[:, 0], np.cumsum(pnl), atol=1e-9)
    np.testing.assert_allclose(out[:, 3], np.abs(dpos) * (o * 2e-4 + fill * 5e-4),
                               atol=1e-12)


def test_slippage_units():
    from screamer import BacktestOHLCTarget
    o, h, l, c = _bars()
    t = np.array([1., 1.])
    frac = BacktestOHLCTarget(slippage=1e-3)(t, o, h, l, c)
    np.testing.assert_allclose(frac[1, 3], 101 * 1e-3)
    ticks = BacktestOHLCTarget(slippage=2, slippage_unit="ticks", tick_size=0.01)(t, o, h, l, c)
    np.testing.assert_allclose(ticks[1, 3], 0.02)
    price = BacktestOHLCTarget(slippage=0.03, slippage_unit="price")(t, o, h, l, c)
    np.testing.assert_allclose(price[1, 3], 0.03)
    # a fractional offset rounds up to whole ticks: 101 * 1e-4 = 0.0101 -> 0.02
    grid = BacktestOHLCTarget(slippage=1e-4, tick_size=0.01)(t, o, h, l, c)
    np.testing.assert_allclose(grid[1, 3], 0.02)


def test_sell_slippage_is_adverse():
    from screamer import BacktestOHLCTarget
    o, h, l, c = _bars()
    out = BacktestOHLCTarget(slippage=0.05, slippage_unit="price")(
        np.array([-1., -1.]), o, h, l, c)
    np.testing.assert_allclose(out[1, 3], 0.05)          # sold at 100.95, cost 0.05
    np.testing.assert_allclose(out[1, 0], 100.95 - 101.5)


def test_tick_size_alone_does_not_slip():
    from screamer import BacktestOHLCTarget
    o, h, l, c = _bars()
    out = BacktestOHLCTarget(tick_size=0.01)(np.array([1., 1.]), o, h, l, c)
    np.testing.assert_allclose(out[1, 3], 0.0)


def test_slippage_validation():
    from screamer import BacktestOHLCTarget
    with pytest.raises(ValueError):
        BacktestOHLCTarget(slippage=-1e-4)
    with pytest.raises(ValueError):
        BacktestOHLCTarget(slippage=1, slippage_unit="bps")
    with pytest.raises(ValueError):
        BacktestOHLCTarget(slippage=1, slippage_unit="ticks")        # no tick_size
    with pytest.raises(ValueError):
        BacktestOHLCTarget(fill_at="vwap")
    with pytest.raises(ValueError):
        BacktestOHLCTarget(delay=-1)
    with pytest.raises(ValueError):
        BacktestOHLCTarget(delay=0)                                   # needs fill_at="close"


@pytest.mark.parametrize("fill_at,ref", [
    ("open", 101.0), ("close", 101.5),
    ("typical", (102 + 100.6 + 101.5) / 3), ("ohlc4", (101 + 102 + 100.6 + 101.5) / 4)])
def test_fill_at_reference(fill_at, ref):
    from screamer import BacktestOHLCTarget
    o, h, l, c = _bars()
    out = BacktestOHLCTarget(fill_at=fill_at)(np.array([1., 1.]), o, h, l, c)
    np.testing.assert_allclose(out[1, 0], 101.5 - ref)    # bought at ref, marked at close
    np.testing.assert_allclose(out[1, 3], 0.0)


def test_delay_zero_is_market_on_close_of_signal_bar():
    from screamer import BacktestOHLCTarget
    o, h, l, c = _bars()
    out = BacktestOHLCTarget(fill_at="close", delay=0)(np.array([1., 1.]), o, h, l, c)
    np.testing.assert_allclose(out[:, 2], [1.0, 1.0])     # in on bar 0
    np.testing.assert_allclose(out[:, 0], [0.0, 1.5])     # bought 100 close, marked 101.5


def test_delay_shifts_execution():
    from screamer import BacktestOHLCTarget
    n = 6
    p = np.arange(100., 100. + n)
    tgt = np.array([1., 0., -1., np.nan, 2., 2.])
    out = BacktestOHLCTarget(delay=3)(tgt, p, p + 1, p - 1, p)
    np.testing.assert_allclose(out[:, 2], [0, 0, 0, 1, 0, -1])
    # a NaN target is "no order": the position holds
    out2 = BacktestOHLCTarget(delay=1)(tgt, p, p + 1, p - 1, p)
    np.testing.assert_allclose(out2[:, 2], [0, 1, 0, -1, -1, 2])


@pytest.mark.parametrize("kw", [
    {}, {"delay": 3}, {"fill_at": "close", "delay": 0},
    {"fill_at": "typical", "slippage": 1e-3, "tick_size": 0.01}])
def test_ohlc_target_stream_equals_batch_and_causal(kw):
    from screamer import BacktestOHLCTarget
    rng = np.random.default_rng(2); n = 120
    c = 100 + np.cumsum(rng.standard_normal(n) * 0.3)
    o = c + rng.standard_normal(n) * 0.1
    h = np.maximum(o, c) + 0.2; l = np.minimum(o, c) - 0.2
    tgt = np.round(rng.standard_normal(n))
    op = BacktestOHLCTarget(**kw)
    stream = np.array([op(tgt[i], o[i], h[i], l[i], c[i]) for i in range(n)])
    batch = BacktestOHLCTarget(**kw)(tgt, o, h, l, c)
    np.testing.assert_allclose(stream, batch)
    k = 60                                          # change the future: past rows hold
    tgt2 = tgt.copy(); tgt2[k:] = -tgt2[k:]
    c2 = c.copy(); c2[k:] += 5
    batch2 = BacktestOHLCTarget(**kw)(tgt2, o, h, l, c2)
    np.testing.assert_allclose(batch2[:k], batch[:k])


def test_ohlc_orders_maker_cost_is_fee_only():
    from screamer import BacktestOHLCOrders
    out = BacktestOHLCOrders(maker_fee=1e-4)(
        np.array([99.]), np.array([1.]), np.array([np.nan]), np.array([0.]),
        np.array([100.]), np.array([101.]), np.array([98.]), np.array([100.]))
    np.testing.assert_allclose(out[0, 3], 99 * 1e-4)
    np.testing.assert_allclose(out[0, 0], 1.0 - 99 * 1e-4)


def test_price_target_slippage_on_top_of_spread():
    from screamer import BacktestPriceTarget
    out = BacktestPriceTarget(spread=0.002, slippage=1, slippage_unit="ticks",
                              tick_size=0.01)(np.array([1.]), np.array([100.]))
    np.testing.assert_allclose(out[0, 3], 0.1 + 0.01)    # half-spread 0.1 + one tick


def test_trades_and_l1_target_slippage():
    from screamer import BacktestTradesTarget, BacktestL1Target
    tr = BacktestTradesTarget(slippage=0.02, slippage_unit="price")(
        np.array([1.]), np.array([100.]), np.array([5.]))
    np.testing.assert_allclose(tr[0, 3], 0.02)
    l1 = BacktestL1Target(slippage=0.02, slippage_unit="price")(
        np.array([1.]), np.array([99.9]), np.array([100.1]), np.array([5.]), np.array([5.]))
    np.testing.assert_allclose(l1[0, 3], 0.1 + 0.02)     # half spread vs mid + slippage


# --- BacktestStats and annualization -----------------------------------------

def _stats_case():
    # flat, long 2 steps, flat, short 1 step, flip long 2 steps, flat
    pos = np.array([0, 1, 1, 0, -1, 1, 1, 0.])
    pnl = np.array([0, 1, 2, -1, 3, -2, 1, 1.])
    return np.cumsum(pnl), pnl, pos, np.zeros(8)


def test_backtest_stats_hand_computed():
    from screamer import BacktestStats
    out = np.asarray(BacktestStats()(*_stats_case()))
    exposure, rt, win, pf, avg_win, avg_loss, hold, sortino, calmar = out[-1]
    assert exposure == pytest.approx(5 / 8)
    assert rt == 3                                   # trades of +2, +1, +2
    assert win == 1.0 and np.isnan(pf) and np.isnan(avg_loss)
    assert avg_win == pytest.approx(5 / 3)
    assert hold == pytest.approx(5 / 3)              # held 2, 1, 2 steps
    assert sortino == pytest.approx(0.625 / np.sqrt(5 / 8))
    assert calmar == pytest.approx(0.625 / 2)


def test_backtest_stats_losing_trades_and_profit_factor():
    from screamer import BacktestStats
    pos = np.array([1, 1, 0, -1, 0, 1, 0.])
    pnl = np.array([1, 2, 0, -4, 0, -1, 0.5])          # trades: +3, -4, -0.5
    out = np.asarray(BacktestStats()(np.cumsum(pnl), pnl, pos, np.zeros(7)))[-1]
    assert out[1] == 3 and out[2] == pytest.approx(1 / 3)
    assert out[3] == pytest.approx(3 / 4.5)
    assert out[4] == pytest.approx(3) and out[5] == pytest.approx(-2.25)


def test_backtest_stats_open_trade_not_counted_and_nan_skip():
    from screamer import BacktestStats
    pos = np.array([0, 1, 1, 1.]); pnl = np.array([0, 1, np.nan, 1.])
    out = np.asarray(BacktestStats()(np.cumsum(np.nan_to_num(pnl)), pnl, pos, np.zeros(4)))
    assert np.isnan(out[2]).all()                     # skipped bar
    assert out[-1, 1] == 0                            # still open, no round trip
    assert np.isnan(out[-1, 2])


def test_backtest_stats_stream_equals_batch():
    from screamer import BacktestStats
    rng = np.random.default_rng(3)
    pos = np.round(rng.standard_normal(200)); pnl = rng.standard_normal(200)
    args = (np.cumsum(pnl), pnl, pos, np.abs(rng.standard_normal(200)) * 0.01)
    op = BacktestStats(252)
    stream = np.array([op(*(a[i] for a in args)) for i in range(200)])
    np.testing.assert_allclose(stream, np.asarray(BacktestStats(252)(*args)), equal_nan=True)


def test_periods_per_year_scales_ratios():
    from screamer import BacktestReport, BacktestStats
    args = _stats_case()
    r1 = np.asarray(BacktestReport()(*args))[-1, 5]
    r252 = np.asarray(BacktestReport(periods_per_year=252)(*args))[-1, 5]
    assert r252 == pytest.approx(r1 * np.sqrt(252))
    s1 = np.asarray(BacktestStats()(*args))[-1]
    s252 = np.asarray(BacktestStats(252)(*args))[-1]
    assert s252[7] == pytest.approx(s1[7] * np.sqrt(252))
    assert s252[8] == pytest.approx(s1[8] * 252)
    with pytest.raises(ValueError):
        BacktestReport(periods_per_year=0)
    with pytest.raises(ValueError):
        BacktestStats(-1)


def test_backtest_report_includes_stats():
    from screamer import backtest_report
    run, summ = backtest_report(np.column_stack(_stats_case()), periods_per_year=252)
    for k in ("exposure", "round_trips", "win_rate", "profit_factor", "avg_win",
              "avg_loss", "avg_holding", "sortino", "calmar"):
        assert k in summ and k in run
    assert summ["round_trips"] == 3


# --- Risk exits: stop_loss, take_profit, trailing_stop -------------------------

def _ohlc(rows):
    a = np.array(rows, dtype=float)
    return a[:, 0], a[:, 1], a[:, 2], a[:, 3]


def test_stop_loss_exits_at_level_inside_the_bar():
    from screamer import BacktestOHLCTarget
    o, h, l, c = _ohlc([[100, 100, 100, 100], [100, 101, 99, 100], [99, 100, 94, 96]])
    out = BacktestOHLCTarget(stop_loss=0.05)(np.ones(3), o, h, l, c)
    np.testing.assert_allclose(out[:, 2], [0, 1, 0])
    np.testing.assert_allclose(out[-1, 0], 95 - 100)      # entered 100, stopped at 95


def test_stop_loss_gap_exits_at_the_open():
    from screamer import BacktestOHLCTarget
    o, h, l, c = _ohlc([[100, 100, 100, 100], [100, 101, 99, 100], [90, 92, 88, 91]])
    out = BacktestOHLCTarget(stop_loss=0.05)(np.ones(3), o, h, l, c)
    np.testing.assert_allclose(out[-1, 0], 90 - 100)


def test_take_profit_and_stop_wins_a_tie():
    from screamer import BacktestOHLCTarget
    o, h, l, c = _ohlc([[100, 100, 100, 100], [100, 101, 99, 100], [101, 106, 100, 104]])
    tp = BacktestOHLCTarget(take_profit=0.05)(np.ones(3), o, h, l, c)
    np.testing.assert_allclose(tp[-1, 0], 105 - 100)
    o, h, l, c = _ohlc([[100, 100, 100, 100], [100, 101, 99, 100], [100, 106, 94, 100]])
    both = BacktestOHLCTarget(stop_loss=0.05, take_profit=0.05)(np.ones(3), o, h, l, c)
    np.testing.assert_allclose(both[-1, 0], 95 - 100)       # order unknown: stop first


def test_exit_latch_blocks_same_side_reentry():
    from screamer import BacktestOHLCTarget
    rows = [[100, 100, 100, 100], [100, 101, 99, 100], [99, 100, 94, 96],
            [96, 97, 95, 96], [96, 97, 95, 96], [96, 97, 95, 96], [96, 97, 95, 96]]
    o, h, l, c = _ohlc(rows)
    tgt = np.array([1, 1, 1, 1, 0, 1, 1.])
    out = BacktestOHLCTarget(stop_loss=0.05)(tgt, o, h, l, c)
    # stopped on bar 2; targets of +1 stay blocked until the 0 on bar 4 releases
    np.testing.assert_allclose(out[:, 2], [0, 1, 0, 0, 0, 0, 1])
    rev = BacktestOHLCTarget(stop_loss=0.05)(np.array([1, 1, 1, -1, -1, -1, -1.]), o, h, l, c)
    np.testing.assert_allclose(rev[:, 2], [0, 1, 0, 0, -1, -1, -1])


def test_trailing_stop_follows_the_best_price():
    from screamer import BacktestOHLCTarget
    o, h, l, c = _ohlc([[100, 100, 100, 100], [100, 101, 99, 100],
                        [101, 110, 101, 109], [108, 109, 104, 105]])
    out = BacktestOHLCTarget(trailing_stop=0.05)(np.ones(4), o, h, l, c)
    np.testing.assert_allclose(out[:, 2], [0, 1, 1, 0])
    np.testing.assert_allclose(out[-1, 0], 110 * 0.95 - 100)


def test_short_stop_and_same_bar_exit_after_open_entry():
    from screamer import BacktestOHLCTarget
    o, h, l, c = _ohlc([[100, 100, 100, 100], [100, 106, 99, 104]])
    out = BacktestOHLCTarget(stop_loss=0.05)(np.array([-1, -1.]), o, h, l, c)
    np.testing.assert_allclose(out[:, 2], [0, 0])            # entered at the open, stopped
    np.testing.assert_allclose(out[-1, 0], 100 - 105)
    # fill_at="close" enters at the close, so the same bar's high cannot stop it
    out = BacktestOHLCTarget(stop_loss=0.05, fill_at="close")(np.array([-1, -1.]), o, h, l, c)
    np.testing.assert_allclose(out[:, 2], [0, -1])


def test_exit_pays_slippage_and_fee():
    from screamer import BacktestOHLCTarget
    o, h, l, c = _ohlc([[100, 100, 100, 100], [100, 101, 99, 100], [99, 100, 94, 96]])
    out = BacktestOHLCTarget(stop_loss=0.05, taker_fee=1e-3, slippage=0.1,
                             slippage_unit="price")(np.ones(3), o, h, l, c)
    # entry fill 100.1 (slipped), so the stop is at 100.1 * 0.95; sold 0.1 below it
    exit_fill = 100.1 * 0.95 - 0.1
    np.testing.assert_allclose(out[-1, 3], 0.1 + exit_fill * 1e-3)


@pytest.mark.parametrize("engine", ["price", "trades", "l1"])
def test_event_engines_exit_on_the_crossing_event(engine):
    from screamer import BacktestPriceTarget, BacktestTradesTarget, BacktestL1Target
    p = np.array([100, 98, 94, 93, 99.])
    tgt = np.ones(5)
    if engine == "price":
        out = BacktestPriceTarget(stop_loss=0.05)(tgt, p)
    elif engine == "trades":
        out = BacktestTradesTarget(stop_loss=0.05)(tgt, p, np.ones(5))
    else:
        out = BacktestL1Target(stop_loss=0.05)(tgt, p - 0.0, p + 0.0, np.ones(5), np.ones(5))
    np.testing.assert_allclose(out[:, 2], [1, 1, 0, 0, 0])
    np.testing.assert_allclose(out[-1, 0], 94 - 100)


@pytest.mark.parametrize("kw", [
    {"stop_loss": 0.01}, {"take_profit": 0.01}, {"trailing_stop": 0.01},
    {"stop_loss": 0.01, "take_profit": 0.02, "fill_at": "close"},
    {"trailing_stop": 0.01, "delay": 2, "slippage": 1e-4}])
def test_exits_stream_equals_batch_and_causal(kw):
    from screamer import BacktestOHLCTarget
    rng = np.random.default_rng(4); n = 300
    c = 100 * np.exp(np.cumsum(rng.standard_normal(n) * 0.005))
    o = np.r_[c[0], c[:-1]]
    h = np.maximum(o, c) * (1 + np.abs(rng.standard_normal(n)) * 0.003)
    l = np.minimum(o, c) * (1 - np.abs(rng.standard_normal(n)) * 0.003)
    tgt = np.sign(np.round(rng.standard_normal(n) * 0.6))
    op = BacktestOHLCTarget(**kw)
    stream = np.array([op(tgt[i], o[i], h[i], l[i], c[i]) for i in range(n)])
    batch = BacktestOHLCTarget(**kw)(tgt, o, h, l, c)
    np.testing.assert_allclose(stream, batch)
    assert np.any(np.diff(batch[:, 2]) != 0)
    k = 150
    c2 = c.copy(); c2[k:] *= 1.1; h2 = np.maximum(h, c2); tgt2 = tgt.copy(); tgt2[k:] *= -1
    batch2 = BacktestOHLCTarget(**kw)(tgt2, o, h2, l, c2)
    np.testing.assert_allclose(batch2[:k], batch[:k])


def test_exit_argument_checks():
    from screamer import BacktestOHLCTarget, BacktestPriceTarget
    for kw in ({"stop_loss": -0.1}, {"stop_loss": 1.0}, {"trailing_stop": 1.5},
               {"take_profit": -1}, {"take_profit": float("inf")}):
        with pytest.raises(ValueError):
            BacktestOHLCTarget(**kw)
        with pytest.raises(ValueError):
            BacktestPriceTarget(**kw)


# --- BacktestCarry ---------------------------------------------------------------

def test_carry_charges_the_position_held_into_each_step():
    from screamer import BacktestPriceTarget, BacktestCarry
    price = np.array([100., 101, 102, 103])
    out = BacktestPriceTarget()(np.array([1., 1, -2, -2]), price)
    rate = np.array([0.001, 0.001, 0.001, 0.001])
    adj = np.asarray(BacktestCarry()(*out.T, price, rate))
    # held into steps 1..3: +1, +1, -2 -> carry = pos_prev * price * rate
    carry = np.array([0, 1 * 101, 1 * 102, -2 * 103]) * 0.001
    np.testing.assert_allclose(adj[:, 1], out[:, 1] - carry)
    np.testing.assert_allclose(adj[:, 3], out[:, 3] + carry)
    np.testing.assert_allclose(adj[:, 0], out[:, 0] - np.cumsum(carry))
    np.testing.assert_allclose(adj[:, 2], out[:, 2])


def test_carry_zero_rate_and_nan_row_is_skipped():
    from screamer import BacktestCarry
    eq = np.array([0., 1, np.nan, 2]); pnl = np.array([0., 1, np.nan, 1])
    pos = np.array([1., 1, np.nan, 1]); cost = np.zeros(4)
    price = np.full(4, 10.); rate = np.array([0.1, 0.0, 0.1, 0.1])
    adj = np.asarray(BacktestCarry(multiplier=2.0)(eq, pnl, pos, cost, price, rate))
    np.testing.assert_allclose(adj[1], [1, 1, 1, 0])          # zero rate: no carry
    assert np.isnan(adj[2]).all()                             # skipped row
    np.testing.assert_allclose(adj[3], [2 - 2.0, 1 - 2.0, 1, 2.0])   # 1 * 10 * 0.1 * 2


def test_carry_stream_equals_batch():
    from screamer import BacktestCarry
    rng = np.random.default_rng(5); n = 100
    cols = [np.cumsum(rng.standard_normal(n)), rng.standard_normal(n),
            np.round(rng.standard_normal(n)), np.abs(rng.standard_normal(n)),
            100 + rng.standard_normal(n), rng.standard_normal(n) * 1e-4]
    op = BacktestCarry()
    stream = np.array([op(*(c[i] for c in cols)) for i in range(n)])
    np.testing.assert_allclose(stream, np.asarray(BacktestCarry()(*cols)))
    with pytest.raises(ValueError):
        BacktestCarry(multiplier=0)


# --- regressions from the exit review ------------------------------------------

@pytest.mark.parametrize("engine", ["trades", "l1"])
def test_exit_fires_while_the_target_holds(engine):
    from screamer import BacktestTradesTarget, BacktestL1Target
    p = np.array([100, 98, 94, 99.])
    tgt = np.array([1, np.nan, np.nan, np.nan])            # NaN: hold the position
    if engine == "trades":
        out = BacktestTradesTarget(stop_loss=0.05)(tgt, p, np.ones(4))
    else:
        out = BacktestL1Target(stop_loss=0.05)(tgt, p, p, np.ones(4), np.ones(4))
    np.testing.assert_allclose(out[:, 2], [1, 1, 0, 0])
    np.testing.assert_allclose(out[-1, 0], 94 - 100)


def test_trailing_best_updates_when_the_position_is_resized():
    from screamer import BacktestOHLCTarget
    o, h, l, c = _ohlc([[100, 100, 100, 100], [105, 120, 104, 118], [117, 118, 107, 110]])
    out = BacktestOHLCTarget(trailing_stop=0.10, fill_at="close", delay=0)(
        np.array([2, 1, 1.]), o, h, l, c)
    # bar 1 reduces 2 -> 1 at the close; its high of 120 still sets the trail at 108
    np.testing.assert_allclose(out[:, 2], [2, 1, 0])


def test_exit_price_stays_inside_the_bar():
    from screamer import BacktestOHLCTarget
    # the entry fill slips to 101, so a 0.5% stop (100.495) sits above this bar's high
    o, h, l, c = _ohlc([[100, 100, 100, 100], [100, 100.2, 99.9, 100]])
    out = BacktestOHLCTarget(stop_loss=0.005, slippage=1.0, slippage_unit="price")(
        np.array([1, 1.]), o, h, l, c)
    assert out[1, 2] == 0
    np.testing.assert_allclose(out[1, 0], (100.2 - 1.0) - 101.0)   # exit at the high, slipped


# --- volume limits: BacktestOHLCVTarget and BacktestTradesTarget ---------------

def _flat_bars(n, price=100.0):
    p = np.full(n, price)
    return p, p + 1, p - 1, p


def test_ohlcv_caps_fills_and_carries_the_rest():
    from screamer import BacktestOHLCVTarget
    o, h, l, c = _flat_bars(7)
    vol = np.full(7, 4.0)
    tgt = np.array([10, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])
    out = BacktestOHLCVTarget(max_participation=0.5)(tgt, o, h, l, c, vol)
    # 0.5 * 4 = 2 units per bar, working toward 10 after the target is gone
    np.testing.assert_allclose(out[:, 2], [0, 2, 4, 6, 8, 10, 10])


def test_ohlcv_new_target_replaces_the_working_one():
    from screamer import BacktestOHLCVTarget
    o, h, l, c = _flat_bars(6)
    tgt = np.array([10, np.nan, -1, np.nan, np.nan, np.nan])
    out = BacktestOHLCVTarget(max_participation=1.0)(tgt, o, h, l, c, np.full(6, 3.0))
    np.testing.assert_allclose(out[:, 2], [0, 3, 6, 3, 0, -1])


def test_ohlcv_without_a_binding_cap_matches_ohlc_target():
    from screamer import BacktestOHLCTarget, BacktestOHLCVTarget
    rng = np.random.default_rng(8); n = 200
    c = 100 * np.exp(np.cumsum(rng.standard_normal(n) * 0.01))
    o = np.r_[c[0], c[:-1]]; h = np.maximum(o, c) * 1.002; l = np.minimum(o, c) * 0.998
    tgt = np.round(rng.standard_normal(n) * 2)
    kw = dict(taker_fee=1e-4, slippage=1e-4, stop_loss=0.02)
    a = BacktestOHLCTarget(**kw)(tgt, o, h, l, c)
    b = BacktestOHLCVTarget(max_participation=1.0, **kw)(tgt, o, h, l, c, np.full(n, 1e9))
    np.testing.assert_allclose(a, b)


def test_ohlcv_nan_or_negative_volume_skips_the_bar_and_streams():
    from screamer import BacktestOHLCVTarget
    o, h, l, c = _flat_bars(4)
    out = BacktestOHLCVTarget()(np.array([1., 1, 1, 1]), o, h, l, c,
                                np.array([5, np.nan, -1, 5.]))
    assert np.isnan(out[1]).all() and np.isnan(out[2]).all()
    rng = np.random.default_rng(9); n = 150
    c = 100 * np.exp(np.cumsum(rng.standard_normal(n) * 0.01))
    o = np.r_[c[0], c[:-1]]; h = np.maximum(o, c) * 1.002; l = np.minimum(o, c) * 0.998
    vol = rng.uniform(0.5, 3, n); tgt = np.round(rng.standard_normal(n) * 3)
    op = BacktestOHLCVTarget(max_participation=0.3, size="units")
    stream = np.array([op(tgt[i], o[i], h[i], l[i], c[i], vol[i]) for i in range(n)])
    np.testing.assert_allclose(stream, BacktestOHLCVTarget(max_participation=0.3)(
        tgt, o, h, l, c, vol))
    with pytest.raises(ValueError):
        BacktestOHLCVTarget(max_participation=0)


def test_trades_target_participation_cap():
    from screamer import BacktestTradesTarget
    p = np.full(5, 100.0); size = np.full(5, 2.0)
    tgt = np.array([3, np.nan, np.nan, np.nan, np.nan])
    out = BacktestTradesTarget(max_participation=0.5)(tgt, p, size)
    np.testing.assert_allclose(out[:, 2], [1, 2, 3, 3, 3])
    full = BacktestTradesTarget()(tgt, p, size)                 # default: no cap
    np.testing.assert_allclose(full[:, 2], [3, 3, 3, 3, 3])


# --- sizing: notional and weight targets -------------------------------------

def test_notional_target_fixes_units_until_it_changes():
    from screamer import BacktestOHLCTarget
    o = np.array([100., 100, 125, 125, 80]); h = o + 1; l = o - 1; c = o
    tgt = np.array([1000, 1000, 1000, 2000, 2000.])
    out = BacktestOHLCTarget(size="notional")(tgt, o, h, l, c)
    # 1000 / 100 = 10 units; held at 125; 2000 / 80 = 25 when the target changes
    np.testing.assert_allclose(out[:, 2], [0, 10, 10, 10, 25])


def test_weight_target_compounds_and_rebalances_on_drift():
    from screamer import BacktestPriceTarget
    price = np.array([100., 120, 140, 140])
    w = np.full(4, 0.5)
    plain = BacktestPriceTarget(size="weight", initial_capital=1000)(w, price)
    np.testing.assert_allclose(plain[:, 2], [5, 5, 5, 5])         # no band: keep units
    band = BacktestPriceTarget(size="weight", initial_capital=1000, rebalance_band=0.05)(w, price)
    # at 120 the weight is 600 / 1100 = 0.545 (inside the band); at 140 it is
    # 700 / 1200 = 0.583, so it rebalances to 0.5 * 1200 / 140 units
    np.testing.assert_allclose(band[:3, 2], [5, 5, 0.5 * 1200 / 140])


@pytest.mark.parametrize("engine", ["ohlc", "price", "trades", "l1"])
def test_weight_sizing_on_every_target_engine(engine):
    import screamer as s
    p = np.array([50., 50, 50])
    w = np.array([0.4, 0.4, 0.4])
    kw = dict(size="weight", initial_capital=10000)
    if engine == "ohlc":
        out = s.BacktestOHLCTarget(**kw)(w, p, p, p, p)
        expect = [0, 80, 80]                                        # executes next bar
    elif engine == "price":
        out = s.BacktestPriceTarget(**kw)(w, p); expect = [80, 80, 80]
    elif engine == "trades":
        out = s.BacktestTradesTarget(**kw)(w, p, np.full(3, 1e6)); expect = [80, 80, 80]
    else:
        out = s.BacktestL1Target(**kw)(w, p, p, np.full(3, 1e6), np.full(3, 1e6))
        expect = [80, 80, 80]
    np.testing.assert_allclose(out[:, 2], expect)


def test_sizing_with_exits_and_multiplier():
    from screamer import BacktestOHLCTarget
    o, h, l, c = _ohlc([[100, 100, 100, 100], [100, 101, 99, 100], [99, 100, 94, 96]])
    out = BacktestOHLCTarget(size="notional", multiplier=10, stop_loss=0.05)(
        np.full(3, 5000.), o, h, l, c)
    np.testing.assert_allclose(out[:, 2], [0, 5, 0])               # 5000 / (100 * 10)
    np.testing.assert_allclose(out[-1, 0], 5 * 10 * (95 - 100))


def test_sizing_argument_checks():
    from screamer import BacktestOHLCTarget, BacktestPriceTarget
    with pytest.raises(ValueError):
        BacktestOHLCTarget(size="percent")
    with pytest.raises(ValueError):
        BacktestOHLCTarget(size="weight")                           # needs capital
    with pytest.raises(ValueError):
        BacktestPriceTarget(rebalance_band=0.1)                     # units has no band
    with pytest.raises(ValueError):
        BacktestPriceTarget(size="notional", rebalance_band=0.1)    # band needs capital
