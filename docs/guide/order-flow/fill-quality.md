---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# Measure the effective and realized spread of your fills

`EffectiveSpread` measures what a fill paid against the mid,
`2 * |price - mid|`, and `RealizedSpread` measures what the other side kept
once the mid had moved; the gap between them is the price impact of the trade.

```{code-cell} ipython3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from screamer import EffectiveSpread, Lag, RealizedSpread

trades = pd.read_csv("../../notebooks/data/deribit_btc_perp_6h.csv")
price, tick = trades["price"].to_numpy(), 0.5
# Quote from each trade: a buy fills at the ask, a sell at the bid.
bid = price - tick * (trades["volume"].to_numpy() > 0)
mid = Lag(1)(bid + tick / 2)               # the mid before each trade

effective = EffectiveSpread()(price, mid)
lags = [1, 3, 10, 30, 100, 300]
realized = {lag: RealizedSpread(lag)(price, mid) for lag in lags}
print(f"mean effective spread {np.nanmean(effective):.2f} USD")
```

```{code-cell} ipython3
mean_realized = [np.nanmean(realized[lag]) for lag in lags]
fig, ax = plt.subplots(figsize=(9, 3.5))
ax.axhline(np.nanmean(effective), color="k", ls="--", lw=0.8,
           label="EffectiveSpread")
ax.plot(lags, mean_realized, "o-", color="steelblue", label="RealizedSpread")
ax.axhline(0, color="0.5", lw=0.5)
ax.set_xscale("log")
ax.set_xlabel("trades until the later mid (lag)")
ax.set_ylabel("mean spread (USD)")
ax.legend(loc="lower left", fontsize=8)
plt.tight_layout()
```

The data file holds Deribit BTC perpetual trades, not quotes, so the page
builds the quote from each trade as [Backtest against bid and ask quotes
(L1)](../backtest/quotes.md) does: a buy fills at the ask, a sell at the bid,
and the spread is one tick of 0.5 USD. `Lag(1)` gives each trade the mid built
from the trade before it. Every trade on the tape counts as a fill here; to
measure your own fills, run both functions on the tape and read the rows where
your orders filled.

`EffectiveSpread()(price, mid)` is `2 * |price - mid|`: a fill at the bid or
ask of a one-tick quote gives 0.5 USD, and a fill away from the last quote, as
when an order walks through the book, gives more, which lifts the mean to
1.47 USD. `RealizedSpread(lag)(price, mid)`
is `2 * D * (price - mid)` with the mid taken `lag` trades later and `D = +1`
for a fill above the mid and -1 below it. Its value at a row belongs to the
trade `lag` rows earlier, so it never uses a future price.

One trade later the realized spread is still positive, 0.42 USD, but after 10
trades it is -4.43 USD: the mid kept moving in the direction of the trades, so a
liquidity provider on the other side lost money on them despite earning the
spread. The price impact, effective minus realized, peaks at about 10 USD after
30 to 100 trades.

## Go deeper

- [`EffectiveSpread`](../../functions_micro/EffectiveSpread.md), [`RealizedSpread`](../../functions_micro/RealizedSpread.md): the formulas and the warmup.
- [Measure price impact](price-impact.md): impact per unit of net order flow.
- [Estimate the bid-ask spread from OHLC bars](spread-from-bars.md): the spread when there are no quotes.
- [Toxicity and book notebook](../../notebooks/13-microstructure-toxicity-and-book.ipynb): effective, realized and impact on a simulated book.
