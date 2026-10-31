"""The `cost` column measures each fill against its documented reference.

A resting (maker) fill is measured against its own limit price, so its cost is
its fee alone. A taker fill is measured against the engine's reference price:
the print (trades), the mid (L1), the execution bar's price (OHLC) or the
price (Price). The reference only splits the result between `pnl` and `cost`;
`equity` and `pnl` are those of the 3.9.1 release on the same inputs.
"""
import numpy as np
import pytest

import screamer as s

nan = float("nan")
inf = float("inf")

MULT = 2.0
ORDERS = dict(maker_fee=0.001, taker_fee=0.002, multiplier=MULT,
              maker_fee_per_contract=0.1, taker_fee_per_contract=0.3)
TARGET = dict(taker_fee=0.002, multiplier=MULT, fee_per_contract=0.3)
MAKER = (0.001, 0.1)       # (fee_rate, fee_per_contract) of a maker fill
TAKER = (0.002, 0.3)       # (fee_rate, fee_per_contract) of a taker fill


def _run(engine, x):
    return engine(*x) if isinstance(x, tuple) else engine(np.asarray(x, float))


# id, engine, input, row, dpos, fill, reference, fees, (equity, pnl) of 3.9.1
CASES = [
    # Trades: a resting bid filled by a print through it; a resting ask likewise.
    ("trades-maker-buy-through", lambda: s.BacktestTradesOrders(**ORDERS),
     [[100, 1, nan, 0, 101, 5], [100, 1, nan, 0, 99, 5]],
     1, 1.0, 100.0, 100.0, MAKER, (-2.3, -2.3)),
    ("trades-maker-sell-through", lambda: s.BacktestTradesOrders(**ORDERS),
     [[nan, 0, 101, 1, 100, 5], [nan, 0, 101, 1, 102, 5]],
     1, -1.0, 101.0, 101.0, MAKER, (-2.302, -2.302)),
    ("trades-maker-touch", lambda: s.BacktestTradesOrders(**ORDERS),
     [[100, 3, nan, 0, 101, 5], [100, 3, nan, 0, 100, 2]],
     1, 2.0, 100.0, 100.0, MAKER, (-0.6, -0.6)),
    ("trades-taker-market", lambda: s.BacktestTradesOrders(
        **ORDERS, slippage=0.01, slippage_unit="price"),
     [[inf, 1, nan, 0, 100, 5]],
     0, 1.0, 100.01, 100.0, TAKER, (-0.72004, -0.72004)),
    ("trades-target", lambda: s.BacktestTradesTarget(
        **TARGET, slippage=0.01, slippage_unit="price"),
     [[1, 100, 5]],
     0, 1.0, 100.01, 100.0, TAKER, (-0.72004, -0.72004)),
    # L1: a resting bid run over by the ask; a touch lock; a crossing new bid.
    ("l1-maker-run-over", lambda: s.BacktestL1Orders(**ORDERS),
     [[99, 1, nan, 0, 99.5, 100.5, 5, 5], [99, 1, nan, 0, 97, 98, 5, 5]],
     1, 1.0, 99.0, 99.0, MAKER, (-3.298, -3.298)),
    ("l1-maker-touch-lock", lambda: s.BacktestL1Orders(**ORDERS, fill="touch"),
     [[99, 1, nan, 0, 98, 100, 5, 5], [99, 1, nan, 0, 98, 99, 5, 5]],
     1, 1.0, 99.0, 99.0, MAKER, (-1.298, -1.298)),
    ("l1-maker-sell-run-over", lambda: s.BacktestL1Orders(**ORDERS),
     [[nan, 0, 101, 1, 99.5, 100.5, 5, 5], [nan, 0, 101, 1, 102, 103, 5, 5]],
     1, -1.0, 101.0, 101.0, MAKER, (-3.302, -3.302)),
    ("l1-taker-crossing", lambda: s.BacktestL1Orders(**ORDERS),
     [[102, 1, nan, 0, 100, 101, 5, 5]],
     0, 1.0, 101.0, 100.5, TAKER, (-1.704, -1.704)),
    ("l1-target", lambda: s.BacktestL1Target(**TARGET),
     [[1, 100, 101, 5, 5]],
     0, 1.0, 101.0, 100.5, TAKER, (-1.704, -1.704)),
    # L1 + trades: a print through a resting bid inside the spread (mid != bid),
    # the run-over fallback without a print, and a crossing new bid.
    ("l1trades-maker-print", lambda: s.BacktestL1TradesOrders(**ORDERS),
     [[99, 1, nan, 0, 98, 100.5, 5, 5, nan, nan],
      [99, 1, nan, 0, 98, 100.5, 5, 5, 98.5, 2]],
     1, 1.0, 99.0, 99.0, MAKER, (0.202, 0.202)),
    ("l1trades-maker-run-over", lambda: s.BacktestL1TradesOrders(**ORDERS),
     [[99, 1, nan, 0, 98, 100.5, 5, 5, nan, nan],
      [99, 1, nan, 0, 97, 98.5, 5, 5, nan, nan]],
     1, 1.0, 99.0, 99.0, MAKER, (-2.798, -2.798)),
    ("l1trades-taker-crossing", lambda: s.BacktestL1TradesOrders(**ORDERS),
     [[102, 1, nan, 0, 100, 101, 5, 5, nan, nan]],
     0, 1.0, 101.0, 100.5, TAKER, (-1.704, -1.704)),
    # OHLC: a resting bid reached by the low; a market buy at the open.
    ("ohlc-maker", lambda: s.BacktestOHLCOrders(**ORDERS),
     [[100, 1, nan, 0, 102, 103, 97, 98]],
     0, 1.0, 100.0, 100.0, MAKER, (-4.3, -4.3)),
    ("ohlc-taker-market", lambda: s.BacktestOHLCOrders(
        **ORDERS, slippage=0.01, slippage_unit="price"),
     [[nan, 1, nan, 0, 102, 103, 97, 98]],
     0, 1.0, 102.01, 102.0, TAKER, (-8.72804, -8.72804)),
    ("ohlc-target", lambda: s.BacktestOHLCTarget(
        **TARGET, slippage=0.01, slippage_unit="price"),
     [[1, 100, 101, 99, 100], [1, 102, 103, 97, 98]],
     1, 1.0, 102.01, 102.0, TAKER, (-8.72804, -8.72804)),
    ("ohlcv-target", lambda: s.BacktestOHLCVTarget(
        **TARGET, slippage=0.01, slippage_unit="price"),
     [[1, 100, 101, 99, 100, 10], [1, 102, 103, 97, 98, 10]],
     1, 1.0, 102.01, 102.0, TAKER, (-8.72804, -8.72804)),
    # Price: half the spread on top of the price.
    ("price-target", lambda: s.BacktestPriceTarget(
        spread=0.002, fee=0.002, multiplier=MULT, fee_per_contract=0.3),
     (np.array([1.0]), np.array([100.0])),
     0, 1.0, 100.1, 100.0, TAKER, (-0.9004, -0.9004)),
]


@pytest.mark.parametrize("case", CASES, ids=[c[0] for c in CASES])
def test_cost_is_fill_versus_reference_plus_fees(case):
    _, make, x, row, dpos, fill, ref, (rate, fpc), _ = case
    out = _run(make(), x)
    fees = abs(dpos) * (fpc + fill * MULT * rate)
    assert out[row, 2] - (out[row - 1, 2] if row else 0.0) == pytest.approx(dpos)
    assert out[row, 3] == pytest.approx(dpos * (fill - ref) * MULT + fees, abs=1e-12)


@pytest.mark.parametrize("case", CASES, ids=[c[0] for c in CASES])
def test_equity_and_pnl_match_the_3_9_1_release(case):
    _, make, x, row, *_, (equity, pnl) = case
    out = _run(make(), x)
    assert out[row, 0] == pytest.approx(equity, abs=1e-12)
    assert out[row, 1] == pytest.approx(pnl, abs=1e-12)


@pytest.mark.parametrize("make, x, row", [
    (lambda: s.BacktestTradesOrders(maker_fee=0.001),
     [[99, 1, nan, 0, 101, 5], [99, 1, nan, 0, 98, 5]], 1),
    (lambda: s.BacktestL1Orders(maker_fee=0.001),
     [[99, 1, nan, 0, 99.5, 100.5, 5, 5], [99, 1, nan, 0, 97, 98, 5, 5]], 1),
    (lambda: s.BacktestL1Orders(maker_fee=0.001, fill="touch"),
     [[99, 1, nan, 0, 98, 100, 5, 5], [99, 1, nan, 0, 98, 99, 5, 5]], 1),
    (lambda: s.BacktestL1TradesOrders(maker_fee=0.001),
     [[99, 1, nan, 0, 98, 100.5, 5, 5, nan, nan],
      [99, 1, nan, 0, 98, 100.5, 5, 5, 98.5, 2]], 1),
    (lambda: s.BacktestL1TradesOrders(maker_fee=0.001),
     [[99, 1, nan, 0, 98, 100.5, 5, 5, nan, nan],
      [99, 1, nan, 0, 97, 98.5, 5, 5, nan, nan]], 1),
], ids=["trades-through", "l1-run-over", "l1-touch-lock",
        "l1trades-print", "l1trades-run-over"])
def test_resting_fill_cost_is_its_fee(make, x, row):
    out = make()(np.asarray(x, float))
    assert out[row, 2] == 1.0
    assert out[row, 3] == pytest.approx(0.099, abs=1e-12)    # 1 bought at 99, 0.1% fee


# Portfolio engines: two assets with their own multiplier and per-contract fee.
PORTFOLIO = dict(fee=0.002, multiplier=[2.0, 3.0], fee_per_contract=[0.3, 0.5],
                 size="units", slippage=0.01, slippage_unit="price",
                 initial_capital=1000.0)
_TARGET = np.array([[1.0, -2.0], [1.0, -2.0], [3.0, 1.0]])
_OPEN = np.array([[100.0, 50.0], [101.0, 49.0], [102.0, 51.0]])


@pytest.mark.parametrize("make, args, ref, last", [
    (lambda: s.BacktestPortfolioTarget(**PORTFOLIO), (_TARGET, _OPEN), _OPEN,
     [(1.8238799999999693, 0.5439199999999795), (-10.168059999999969, -14.508179999999982)]),
    # fill_at="open", delay=1: a target fills at the next bar's open.
    (lambda: s.BacktestPortfolioOHLCTarget(**PORTFOLIO),
     (_TARGET, _OPEN, _OPEN + 2, _OPEN - 2, _OPEN + 1), np.vstack([[nan, nan], _OPEN[1:]]),
     [(3.2759599999999898, 2.0), (-19.647879999999986, -12.0)]),
], ids=["portfolio-target", "portfolio-ohlc-target"])
def test_portfolio_cost_is_fill_versus_reference_plus_fees(make, args, ref, last):
    out = np.asarray(make()(*args))
    mult, fpc = np.array(PORTFOLIO["multiplier"]), np.array(PORTFOLIO["fee_per_contract"])
    dpos = np.diff(out[:, :, 2], axis=0, prepend=0.0)
    traded = dpos != 0
    assert traded.sum() >= 2
    fill = ref + np.sign(dpos) * 0.01
    want = dpos * (fill - ref) * mult + np.abs(dpos) * (fpc + fill * mult * 0.002)
    np.testing.assert_allclose(out[:, :, 3][traded], want[traded], atol=1e-12)
    # equity and pnl of the last row are those of the 3.9.1 release
    np.testing.assert_allclose(out[-1, :, :2], last, atol=1e-12)
