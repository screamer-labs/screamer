---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# Backtest on tick trades instead of bars

`BacktestTradesTarget` trades a target position against each print of the
trade tape as a market order, and `BacktestTradesOrders` rests a bid and an ask
that later prints fill; both run on the raw trades, without bars.

```{code-cell} ipython3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from screamer import (EwMean, Sign, Lag, FillNa, BacktestTradesTarget,
                      BacktestTradesOrders, backtest_report)

trades = pd.read_csv("../../notebooks/data/deribit_btc_perp_6h.csv")
price = trades["price"].to_numpy()
size = np.abs(trades["volume"].to_numpy()) / price     # USD to BTC
trend = Sign()(EwMean(span=100)(price) - EwMean(span=600)(price))
side = FillNa(0.0)(Lag(1)(trend))                      # known before the print
last = FillNa(price[0])(Lag(1)(price))                 # the previous print

q = 0.1                                                # BTC
taker = BacktestTradesTarget(taker_fee=5e-4)(q * side, price, size)
maker = BacktestTradesOrders(min_position=-q, max_position=q)(
    last, 2 * q * (side > 0), last, 2 * q * (side < 0), price, size)
pd.DataFrame({name: backtest_report(out)[1] for name, out in
              {"market orders": taker, "resting orders": maker}.items()}
             ).loc[["total_pnl", "num_trades"]].round(1)
```

```{code-cell} ipython3
t = pd.to_datetime(trades["timestamp"], unit="ms")
fig, ax = plt.subplots(figsize=(9, 3.5))
ax.plot(t, taker[:, 0], color="crimson", label="market orders")
ax.plot(t, maker[:, 0], color="steelblue", label="resting orders")
ax.axhline(0, color="k", lw=0.5)
ax.set_ylabel("profit (USD)")
ax.legend(loc="lower left", fontsize=8)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
plt.tight_layout()
```

The tape is six hours of Deribit BTC-perpetual trades, 4,093 prints. The file
gives each trade's size in USD, signed by the side that started it; the page
turns it into BTC. The signal compares moving averages over 100 and 600
prints and asks for a position of 0.1 BTC long or short. Both engines act on
the print they are given, so the signal is lagged one print with `Lag`: the
order on each print uses only the prints before it.

`BacktestTradesTarget` trades the difference between the target and the
position at the print's price, as a market order that pays `taker_fee`, here
0.05%. `BacktestTradesOrders` instead rests a bid when the signal is long, or
an ask when it is short, at the price of the previous print, with size 0.2 BTC
and the position held between -0.1 and 0.1. With the default `fill="touch"`, a
print at the bid price fills `participation_ratio` times the print's size, up
to the bid's size and the room left; a print below the bid fills the bid's
whole size, up to the room left. Both fill at the bid price, as maker fills
that pay `maker_fee`, here 0, and the ask is the mirror image.

These maker fills apply to a bid that rests at the same price as on the print
before. A bid arrives when it is new or its price changed, and an arriving bid
above the print is marketable: it fills its whole size at the print, as a
taker that pays `taker_fee`, here 0. The bid here follows the previous print,
so it re-prices whenever the price moves, and 8 of the 77 fills are of this
kind.

The market orders lose more, mostly to their fees: 0.05% of 0.1 BTC is about
1.5 USD, and a reversal trades 0.2 BTC. The resting orders pay no fee, but
they fill only when a print comes down to the bid or up to the ask, which is
often just as the price moves against the new position.

## Go deeper

- [`BacktestTradesTarget`](../../functions_fin/BacktestTradesTarget.md): `max_participation` of each print, sizing, and risk exits.
- [`BacktestTradesOrders`](../../functions_fin/BacktestTradesOrders.md): the fill rules on the tape.
- [Backtest against bid and ask quotes (L1)](quotes.md): the same two order types against a bid and an ask.
- [Event-driven backtests notebook](../../notebooks/15-event-driven-backtests.ipynb): a mean-reverting maker on the tape, and quotes with trades.
- [Build bars from trades](../get-data-in/bars-from-ticks.md): when bars are enough.
