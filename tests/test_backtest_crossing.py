"""Orders that cross the market on arrival fill as takers at the market.

A side arrives on an event when it was not quoted on the previous event or its
price changed; a size change alone is not an arrival. An arriving order that
crosses the market fills at the best market price, capped at its limit, and
pays the taker fee. A resting, unchanged order keeps the maker behaviour. On
OHLC bars each quote is placed at the open, so a limit through the open fills
at the open as a taker and every fill lies within the bar.

Columns of every engine's output: equity, pnl, position, cost.
"""
import numpy as np
import pytest

import screamer as s

nan = float("nan")
inf = float("inf")
FEES = dict(maker_fee=0.001, taker_fee=0.002)


def _fill(row, close, prev_pos=0.0, mark_pnl=0.0):
    """Fill price recovered from one output row of a fee-free run."""
    dpos = row[2] - prev_pos
    return close - (row[1] - mark_pnl) / dpos


def _taker_cost(dpos, fill, reference):
    return dpos * (fill - reference) + abs(dpos) * fill * FEES["taker_fee"]


def _maker_cost(dpos, fill):
    return abs(dpos) * fill * FEES["maker_fee"]


# ---------------------------------------------------------------- OHLC orders

@pytest.mark.parametrize("fill", ["touch", "breach"])
def test_ohlc_bid_above_bar_fills_at_open_as_taker(fill):
    out = s.BacktestOHLCOrders(fill=fill, **FEES)(
        np.array([[110, 1, nan, 0, 100, 105, 95, 100]], float))
    assert out[0, 2] == 1.0
    np.testing.assert_allclose(out[0, 3], _taker_cost(1, 100, 100))   # fee at 100
    np.testing.assert_allclose(out[0, 1], -100 * 0.002)                # bought at 100, closed at 100


def test_ohlc_gap_through_resting_bid_fills_at_open_as_taker():
    out = s.BacktestOHLCOrders(**FEES)(
        np.array([[100, 1, nan, 0, 95, 97, 93, 96]], float))
    assert out[0, 2] == 1.0
    np.testing.assert_allclose(out[0, 3], _taker_cost(1, 95, 95))
    np.testing.assert_allclose(out[0, 1], 96 - 95 - 95 * 0.002)


def test_ohlc_ask_below_bar_and_gap_up_fill_at_open_as_taker():
    out = s.BacktestOHLCOrders(**FEES)(
        np.array([[nan, 0, 90, 1, 100, 105, 95, 100]], float))
    assert out[0, 2] == -1.0
    np.testing.assert_allclose(out[0, 3], _taker_cost(-1, 100, 100))
    out = s.BacktestOHLCOrders(**FEES)(
        np.array([[nan, 0, 100, 1, 105, 107, 103, 104]], float))
    np.testing.assert_allclose(out[0, 1], -(104 - 105) - 105 * 0.002)


def test_ohlc_crossing_slippage_is_capped_at_the_limit():
    eng = dict(slippage=1.0, slippage_unit="price")
    out = s.BacktestOHLCOrders(**eng)(np.array([[100.5, 1, nan, 0, 100, 105, 95, 100]], float))
    np.testing.assert_allclose(_fill(out[0], 100), 100.5)              # min(100.5, 100 + 1)
    out = s.BacktestOHLCOrders(**eng)(np.array([[nan, 0, 99.5, 1, 100, 105, 95, 100]], float))
    np.testing.assert_allclose(_fill(out[0], 100), 99.5)               # max(99.5, 100 - 1)
    out = s.BacktestOHLCOrders(**eng)(np.array([[110, 1, nan, 0, 100, 105, 95, 100]], float))
    np.testing.assert_allclose(_fill(out[0], 100), 101.0)              # slippage below the limit


def test_ohlc_resting_limit_inside_the_bar_is_a_maker_at_its_limit():
    out = s.BacktestOHLCOrders(**FEES)(np.array([[99, 1, nan, 0, 100, 105, 95, 100]], float))
    np.testing.assert_allclose(out[0, 3], _maker_cost(1, 99))
    np.testing.assert_allclose(out[0, 1], 100 - 99 - 99 * 0.001)
    out = s.BacktestOHLCOrders(fill="breach", **FEES)(   # open == bid: not through, breach needs low < bid
        np.array([[100, 1, nan, 0, 100, 105, 100, 102]], float))
    assert out[0, 2] == 0.0


def test_ohlc_nan_bar_and_never_fill_limits_are_unchanged():
    out = s.BacktestOHLCOrders()(np.array([[110, 1, nan, 0, nan, 105, 95, 100],
                                           [-inf, 1, inf, 1, 100, 105, 95, 100]], float))
    assert np.isnan(out[0]).all()
    assert out[1, 2] == 0.0


@pytest.mark.parametrize("fill", ["touch", "breach"])
def test_ohlc_every_fill_lies_within_the_bar(fill):
    rng = np.random.default_rng(7)
    n = 400
    o = 100 + rng.normal(0, 2, n)
    c = 100 + rng.normal(0, 2, n)
    h = np.maximum(o, c) + rng.exponential(1, n)
    l = np.minimum(o, c) - rng.exponential(1, n)
    limit = 100 + rng.normal(0, 4, n)
    buy = rng.random(n) < 0.5
    fills = 0
    for i in range(n):
        row = [limit[i], 1, nan, 0] if buy[i] else [nan, 0, limit[i], 1]
        out = s.BacktestOHLCOrders(fill=fill)(np.array([row + [o[i], h[i], l[i], c[i]]], float))
        if out[0, 2] != 0.0:
            fills += 1
            f = _fill(out[0], c[i])
            assert l[i] - 1e-9 <= f <= h[i] + 1e-9
    assert fills > 100


# -------------------------------------------------------------- Trades orders

def test_trades_new_bid_above_print_fills_at_print_as_taker():
    out = s.BacktestTradesOrders(**FEES)(np.array([[102, 1, nan, 0, 100, 1]], float))
    assert out[0, 2] == 1.0
    np.testing.assert_allclose(out[0, 3], _taker_cost(1, 100, 100))
    np.testing.assert_allclose(out[0, 1], -100 * 0.002)


def test_trades_repriced_bid_is_an_arrival():
    out = s.BacktestTradesOrders(**FEES)(
        np.array([[99, 1, nan, 0, 100, 1], [102, 1, nan, 0, 100, 1]], float))
    assert out[1, 2] == 1.0
    np.testing.assert_allclose(out[1, 3], _taker_cost(1, 100, 100))


def test_trades_resting_bid_through_print_is_a_maker_at_its_limit():
    out = s.BacktestTradesOrders(**FEES)(
        np.array([[100, 1, nan, 0, 101, 1], [100, 1, nan, 0, 99, 1]], float))
    assert out[1, 2] == 1.0
    np.testing.assert_allclose(out[1, 3], _maker_cost(1, 100))
    np.testing.assert_allclose(out[1, 1], 99 - 100 - 100 * 0.001)


def test_trades_size_only_change_is_not_an_arrival():
    out = s.BacktestTradesOrders(**FEES)(
        np.array([[100, 1, nan, 0, 101, 1], [100, 2, nan, 0, 99, 1]], float))
    assert out[1, 2] == 2.0
    np.testing.assert_allclose(out[1, 3], _maker_cost(2, 100))


def test_trades_nan_then_crossing_bid_is_a_taker():
    out = s.BacktestTradesOrders(**FEES)(
        np.array([[nan, 0, nan, 0, 101, 1], [102, 1, nan, 0, 100, 1]], float))
    np.testing.assert_allclose(out[1, 3], _taker_cost(1, 100, 100))


def test_trades_new_bid_at_the_print_stays_resting():
    out = s.BacktestTradesOrders(participation_ratio=0.5, **FEES)(
        np.array([[100, 1, nan, 0, 100, 1]], float))
    assert out[0, 2] == 0.5                                            # touch partial, maker
    np.testing.assert_allclose(out[0, 3], _maker_cost(0.5, 100))
    out = s.BacktestTradesOrders(fill="breach")(np.array([[100, 1, nan, 0, 100, 1]], float))
    assert out[0, 2] == 0.0


def test_trades_new_ask_below_print_fills_at_print_as_taker():
    out = s.BacktestTradesOrders(**FEES)(np.array([[nan, 0, 98, 1, 100, 1]], float))
    assert out[0, 2] == -1.0
    np.testing.assert_allclose(out[0, 3], _taker_cost(-1, 100, 100))


def test_trades_crossing_slippage_is_capped_at_the_limit():
    eng = dict(slippage=1.0, slippage_unit="price")
    out = s.BacktestTradesOrders(**eng)(np.array([[100.5, 1, nan, 0, 100, 1]], float))
    np.testing.assert_allclose(_fill(out[0], 100), 100.5)
    out = s.BacktestTradesOrders(**eng)(np.array([[nan, 0, 99.5, 1, 100, 1]], float))
    np.testing.assert_allclose(_fill(out[0], 100), 99.5)


def test_trades_order_without_room_rearrives():
    # Long 1 at max_position 1: the bid cannot work. Once the ask sells, the bid
    # re-enters the book on the next event and is marketable against print 100.
    out = s.BacktestTradesOrders(max_position=1, **FEES)(
        np.array([[102, 1, nan, 0, 100, 1],
                  [102, 1, 103, 1, 104, 1],
                  [102, 1, nan, 0, 100, 1]], float))
    np.testing.assert_allclose(out[:, 2], [1, 0, 1])
    np.testing.assert_allclose(out[2, 3], _taker_cost(1, 100, 100))


# ------------------------------------------------------------ L1 / L1Trades

def _l1(rows, engine):
    rows = [r + [nan, nan] if engine is s.BacktestL1TradesOrders and len(r) == 8 else r
            for r in rows]
    return np.array(rows, float)


L1_ENGINES = [s.BacktestL1Orders, s.BacktestL1TradesOrders]


@pytest.mark.parametrize("engine", L1_ENGINES)
@pytest.mark.parametrize("fill", ["touch", "breach"])
def test_l1_repriced_bid_crossing_fills_at_the_ask_as_taker(engine, fill):
    out = engine(fill=fill, **FEES)(_l1([[99, 1, nan, 0, 100, 101, 5, 5],
                                         [102, 1, nan, 0, 100, 101, 5, 5]], engine))
    assert out[1, 2] == 1.0
    np.testing.assert_allclose(out[1, 3], _taker_cost(1, 101, 100.5))


def test_l1trades_repriced_bid_is_a_taker_regardless_of_print():
    eng = s.BacktestL1TradesOrders(**FEES)
    out = eng(np.array([[99, 1, nan, 0, 100, 101, 5, 5, nan, nan],
                        [102, 1, nan, 0, 100, 101, 5, 5, 100, 1]], float))
    np.testing.assert_allclose(out[1, 3], _taker_cost(1, 101, 100.5))


@pytest.mark.parametrize("engine", L1_ENGINES)
@pytest.mark.parametrize("fill", ["touch", "breach"])
def test_l1_new_bid_equal_to_ask_is_a_taker(engine, fill):
    out = engine(fill=fill, **FEES)(_l1([[101, 1, nan, 0, 100, 101, 5, 5]], engine))
    assert out[0, 2] == 1.0
    np.testing.assert_allclose(out[0, 3], _taker_cost(1, 101, 100.5))
    out = engine(fill=fill, **FEES)(_l1([[nan, 0, 100, 1, 100, 101, 5, 5]], engine))
    assert out[0, 2] == -1.0
    np.testing.assert_allclose(out[0, 3], _taker_cost(-1, 100, 100.5))


def test_l1_resting_lock_is_a_maker_partial():
    out = s.BacktestL1Orders(fill="touch", participation_ratio=0.5, **FEES)(
        np.array([[100, 10, nan, 0, 99, 101, 5, 8],
                  [100, 10, nan, 0, 99, 100, 5, 8]], float))
    assert out[1, 2] == 4.0
    np.testing.assert_allclose(out[1, 3], _maker_cost(4, 100))


def test_l1trades_resting_lock_fills_from_the_print_as_maker():
    out = s.BacktestL1TradesOrders(participation_ratio=0.5, **FEES)(
        np.array([[100, 10, nan, 0, 99, 101, 5, 8, nan, nan],
                  [100, 10, nan, 0, 99, 100, 5, 8, 100, 4]], float))
    assert out[1, 2] == 2.0
    np.testing.assert_allclose(out[1, 3], _maker_cost(2, 100))


@pytest.mark.parametrize("engine", L1_ENGINES)
def test_l1_walk_and_slippage_stop_at_the_limit(engine):
    # 10 lots against ask 101 x 4, tick 1: 4 at 101, 6 walk to min(102, 101.5).
    out = engine(tick_size=1.0)(_l1([[101.5, 10, nan, 0, 100, 101, 5, 4]], engine))
    np.testing.assert_allclose(_fill(out[0], 100.5), (4 * 101 + 6 * 101.5) / 10)
    out = engine(tick_size=1.0, slippage=1.0, slippage_unit="price")(
        _l1([[101.5, 10, nan, 0, 100, 101, 5, 4]], engine))
    np.testing.assert_allclose(_fill(out[0], 100.5), 101.5)
    out = engine(tick_size=1.0, slippage=1.0, slippage_unit="price")(
        _l1([[nan, 0, 99.5, 10, 100, 101, 4, 5]], engine))
    np.testing.assert_allclose(_fill(out[0], 100.5), 99.5)
    out = engine(tick_size=1.0)(_l1([[inf, 10, nan, 0, 100, 101, 5, 4]], engine))
    np.testing.assert_allclose(_fill(out[0], 100.5), (4 * 101 + 6 * 102) / 10)   # +inf walk uncapped


@pytest.mark.parametrize("engine", L1_ENGINES)
def test_l1_nan_then_crossing_is_a_taker(engine):
    out = engine(**FEES)(_l1([[nan, 1, nan, 0, 100, 101, 5, 5],
                              [102, 1, nan, 0, 100, 101, 5, 5]], engine))
    np.testing.assert_allclose(out[1, 3], _taker_cost(1, 101, 100.5))


@pytest.mark.parametrize("engine", L1_ENGINES)
def test_l1_size_only_change_is_not_an_arrival(engine):
    out = engine(**FEES)(_l1([[99, 1, nan, 0, 100, 101, 5, 5],
                              [99, 2, nan, 0, 97, 98, 5, 5]], engine))
    assert out[1, 2] == 2.0
    np.testing.assert_allclose(out[1, 3], _maker_cost(2, 99))          # run over: maker at 99
    out = engine(fill="touch", participation_ratio=0.5, **FEES)(
        _l1([[100, 1, nan, 0, 99, 101, 5, 8], [100, 2, nan, 0, 99, 100, 5, 8]], engine))
    if engine is s.BacktestL1Orders:
        assert out[1, 2] == 2.0                                        # lock partial, maker
        np.testing.assert_allclose(out[1, 3], _maker_cost(2, 100))
    else:
        assert out[1, 2] == 0.0                                        # resting: waits for a print


@pytest.mark.parametrize("engine", L1_ENGINES)
def test_l1_resting_side_switched_to_market_is_a_taker(engine):
    out = engine(tick_size=1.0, **FEES)(_l1([[99, 1, nan, 0, 100, 101, 5, 5],
                                             [inf, 1, nan, 0, 100, 101, 5, 5]], engine))
    assert out[1, 2] == 1.0
    np.testing.assert_allclose(out[1, 3], _taker_cost(1, 101, 100.5))
    assert np.isfinite(out[1]).all()
    out = engine(tick_size=1.0, **FEES)(_l1([[nan, 0, 102, 1, 100, 101, 5, 5],
                                             [nan, 0, -inf, 1, 100, 101, 5, 5]], engine))
    assert out[1, 2] == -1.0
    np.testing.assert_allclose(out[1, 3], _taker_cost(-1, 100, 100.5))


@pytest.mark.parametrize("engine", L1_ENGINES)
def test_l1_repriced_ask_crossing_is_a_taker(engine):
    out = engine(**FEES)(_l1([[nan, 0, 102, 1, 100, 101, 5, 5],
                              [nan, 0, 99, 1, 100, 101, 5, 5]], engine))
    np.testing.assert_allclose(out[1, 3], _taker_cost(-1, 100, 100.5))


# ------------------------------------------------ streaming == batch, reset()

def _random_inputs(kind, n=300, seed=3):
    rng = np.random.default_rng(seed)
    mid = 100 + np.cumsum(rng.normal(0, 0.3, n))
    def quote(offset):
        p = np.round(mid + offset + rng.normal(0, 0.4, n), 1)
        hold = rng.random(n) < 0.5                       # repeat the previous price half the time
        for i in range(1, n):
            if hold[i]:
                p[i] = p[i - 1]
        p[rng.random(n) < 0.1] = nan
        return p
    bid, ask = quote(-0.3), quote(0.3)
    size = rng.integers(1, 4, n).astype(float)
    if kind == "ohlc":
        o = mid + rng.normal(0, 0.2, n); c = mid + rng.normal(0, 0.2, n)
        h = np.maximum(o, c) + rng.exponential(0.3, n); l = np.minimum(o, c) - rng.exponential(0.3, n)
        mkt = [o, h, l, c]
    elif kind == "trades":
        mkt = [np.round(mid + rng.normal(0, 0.3, n), 1), size.copy()]
    else:
        mb = np.round(mid - 0.05, 1); ma = mb + 0.1
        mkt = [mb, ma, size.copy(), size.copy()]
        if kind == "l1trades":
            tp = np.where(rng.random(n) < 0.5, np.where(rng.random(n) < 0.5, mb, ma), nan)
            mkt += [tp, np.where(np.isnan(tp), nan, 2.0)]
    return np.column_stack([bid, size, ask, size[::-1].copy()] + mkt)


@pytest.mark.parametrize("kind,factory", [
    ("ohlc", lambda: s.BacktestOHLCOrders(tick_size=0.1, **FEES)),
    ("trades", lambda: s.BacktestTradesOrders(participation_ratio=0.5, **FEES)),
    ("l1", lambda: s.BacktestL1Orders(fill="touch", tick_size=0.1, **FEES)),
    ("l1", lambda: s.BacktestL1Orders(fill="breach", tick_size=0.1, **FEES)),
    ("l1trades", lambda: s.BacktestL1TradesOrders(tick_size=0.1, **FEES)),
])
def test_streaming_equals_batch_and_reset(kind, factory):
    x = _random_inputs(kind)
    op = factory()
    stream = np.array([op(*map(float, row)) for row in x])
    op.reset()
    again = op(x)
    batch = factory()(x)
    np.testing.assert_array_equal(np.nan_to_num(stream), np.nan_to_num(batch))
    np.testing.assert_array_equal(np.nan_to_num(again), np.nan_to_num(batch))
    assert np.count_nonzero(np.diff(np.nan_to_num(batch[:, 2]))) > 20


# ------------------------------------- a fully filled order no longer exists

def test_trades_fully_filled_bid_requoted_is_an_arrival():
    # bid 100 rests above print 101, then prints 99, 97, 95: the first sweep is a
    # maker fill at 100; the order is gone, so the same quote on the next prints
    # is a new order through the market: a taker at each print.
    out = s.BacktestTradesOrders(**FEES)(
        np.array([[100, 1, nan, 0, p, 1] for p in (101, 99, 97, 95)], float))
    np.testing.assert_allclose(out[:, 2], [0, 1, 2, 3])
    np.testing.assert_allclose(out[1, 3], _maker_cost(1, 100))
    np.testing.assert_allclose(out[2, 3], _taker_cost(1, 97, 97))
    np.testing.assert_allclose(out[3, 3], _taker_cost(1, 95, 95))


def test_trades_touch_partial_fill_keeps_the_order_resting():
    out = s.BacktestTradesOrders(participation_ratio=0.5, **FEES)(
        np.array([[100, 1, nan, 0, 101, 1], [100, 1, nan, 0, 100, 1],
                  [100, 1, nan, 0, 99, 1]], float))
    np.testing.assert_allclose(out[:, 2], [0, 0.5, 1.5])
    np.testing.assert_allclose(out[2, 3], _maker_cost(1, 100))      # still resting: maker


@pytest.mark.parametrize("engine", L1_ENGINES)
@pytest.mark.parametrize("fill", ["touch", "breach"])
def test_l1_fully_filled_order_requoted_at_the_ask_is_a_taker(engine, fill):
    # bid 100 rests, is run over on event 1 (maker at 100), and the same quote
    # meets the ask at 100 on event 2: a new order, so a taker.
    out = engine(fill=fill, **FEES)(_l1([[100, 1, nan, 0, 99, 101, 5, 5],
                                         [100, 1, nan, 0, 98, 99, 5, 5],
                                         [100, 1, nan, 0, 99, 100, 5, 5]], engine))
    np.testing.assert_allclose(out[:, 2], [0, 1, 2])
    np.testing.assert_allclose(out[1, 3], _maker_cost(1, 100))
    np.testing.assert_allclose(out[2, 3], _taker_cost(1, 100, 99.5))


def test_l1trades_print_fully_filled_order_requoted_at_the_ask_is_a_taker():
    out = s.BacktestL1TradesOrders(**FEES)(
        np.array([[100, 1, nan, 0, 99, 101, 5, 5, nan, nan],
                  [100, 1, nan, 0, 99, 101, 5, 5, 99.5, 5],
                  [100, 1, nan, 0, 99, 100, 5, 5, nan, nan]], float))
    np.testing.assert_allclose(out[:, 2], [0, 1, 2])
    np.testing.assert_allclose(out[1, 3], _maker_cost(1, 100))
    np.testing.assert_allclose(out[2, 3], _taker_cost(1, 100, 99.5))


def test_l1_touch_lock_that_fills_fully_ends_the_order():
    out = s.BacktestL1Orders(fill="touch", **FEES)(
        np.array([[100, 1, nan, 0, 99, 101, 5, 8],
                  [100, 1, nan, 0, 99, 100, 5, 8],
                  [100, 1, nan, 0, 99, 100, 5, 8]], float))
    np.testing.assert_allclose(out[:, 2], [0, 1, 2])
    np.testing.assert_allclose(out[1, 3], _maker_cost(1, 100))       # lock fills it fully
    np.testing.assert_allclose(out[2, 3], _taker_cost(1, 100, 99.5)) # same quote: new order
