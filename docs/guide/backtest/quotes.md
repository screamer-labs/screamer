---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# Backtest against bid and ask quotes (L1)

`BacktestL1Target` trades a target position against the best bid and ask
(level 1 of the book), buying at the ask and selling at the bid, and
`BacktestL1Orders` rests your own bid and ask and fills them when the market
moves through them.

```{code-cell} ipython3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from screamer import (EwMean, Sign, Lag, FillNa, BacktestL1Target,
                      BacktestL1Orders, backtest_report)

trades = pd.read_csv("../../notebooks/data/deribit_btc_perp_6h.csv")
price, tick = trades["price"].to_numpy(), 0.5
bid = price - tick * (trades["volume"].to_numpy() > 0)  # buys hit the ask
ask, depth = bid + tick, np.ones(len(price))             # 1 BTC on each side
trend = Sign()(EwMean(span=100)(bid) - EwMean(span=600)(bid))
side = FillNa(0.0)(Lag(1)(trend))                        # known before the quote

q = 0.1                                                  # BTC
taker = BacktestL1Target(taker_fee=5e-4, tick_size=tick)(
    q * side, bid, ask, depth, depth)
maker = BacktestL1Orders(taker_fee=5e-4, min_position=-q, max_position=q)(
    FillNa(bid[0])(Lag(1)(bid)), 2 * q * (side > 0),
    FillNa(ask[0])(Lag(1)(ask)), 2 * q * (side < 0), bid, ask, depth, depth)
pd.DataFrame({name: backtest_report(out)[1] for name, out in
              {"cross the spread": taker, "rest at the touch": maker}.items()}
             ).loc[["total_pnl", "num_trades"]].round(1)
```

```{code-cell} ipython3
import matplotlib.dates as mdates

t = pd.to_datetime(trades["timestamp"], unit="ms")
fig, ax = plt.subplots(figsize=(9, 3.5))
ax.plot(t, taker[:, 0], color="crimson", label="cross the spread")
ax.plot(t, maker[:, 0], color="steelblue", label="rest at the touch")
ax.axhline(0, color="k", lw=0.5)
ax.set_ylabel("profit (USD)")
ax.legend(loc="lower left", fontsize=8)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
plt.tight_layout()
```

The data file holds Deribit BTC-perpetual trades, not quotes, so the page
builds a quote from each trade. A trade started by a buyer fills at the ask
and one started by a seller at the bid, and the spread of this market is
usually one tick of 0.5 USD, so a buy at a price gives the ask at that
price and the bid a tick below, and a sell the reverse. The tape does not show
the size on each side, so the page shows 1 BTC on both, more than any order
here. The signal is the moving-average rule of [Backtest on tick trades
instead of bars](tick-trades.md), lagged one event so that each order uses
only the quotes before it.

`BacktestL1Target` buys at the ask and sells at the bid on the same event, and
walks one `tick_size` further for any size beyond what the book shows. It
marks the position to the mid, so each fill pays half the spread on top of
the 0.05% taker fee.

`BacktestL1Orders` rests a bid at the previous event's best bid when the
signal is long, or an ask at the previous best ask when it is short, with size
0.2 BTC and the position held between -0.1 and 0.1. With the default
`fill="breach"`, the bid fills only when the market's ask falls below it
(`market_ask < bid_price`), for its whole size up to the room left, at the bid
price and with `maker_fee`, here 0. `fill="touch"` also fills once when the ask
reaches the bid price, for `participation_ratio` times the size shown there.
A bid that already crosses the market when it arrives (`bid_price >=
market_ask`) is a market order: it fills at the ask and pays `taker_fee`. The
ask side is the mirror image.

Resting at the touch avoids most of the fees and the half spread, but its
fills come when the price has moved through the order. A quote built from
trades only changes when someone trades, so these fills are a rough guide;
with real quotes, prefer the quotes together with the trades, as in the
notebook below.

## Go deeper

- [`BacktestL1Target`](../../functions_fin/BacktestL1Target.md): sizing, risk exits, and the walk past the displayed size.
- [`BacktestL1Orders`](../../functions_fin/BacktestL1Orders.md): the fill rules, and why quotes alone can under- or over-fill.
- [`BacktestL1TradesOrders`](../../functions_fin/BacktestL1TradesOrders.md): quotes and the trade tape together, with fills driven by real trades.
- [Event-driven backtests notebook](../../notebooks/15-event-driven-backtests.ipynb): the bar, tape, and quote engines side by side.
- [Backtest on tick trades instead of bars](tick-trades.md): the same two order types on the trade tape.
