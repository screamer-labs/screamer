---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# Restart a calculation each day, contract or group

`Segment(op)` runs any function from a fresh start on each run of rows with the
same label, such as a day, a futures contract or a regime, so a session VWAP or
a session high starts again when the label changes.

```{code-cell} ipython3
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from screamer import Abs, CumSum, ExpandingMax, ExpandingMin, Segment

trades = pd.read_csv("../../notebooks/data/deribit_btc_perp_6h.csv")
trades.index = pd.to_datetime(trades["timestamp"], unit="ms")
price = trades["price"].to_numpy()
usd = Abs()(trades["volume"].to_numpy())           # trade size in USD
session = trades["timestamp"].to_numpy() // 3_600_000   # hour number

# VWAP: USD traded over BTC traded, both summed since the session began.
trades["vwap"] = (Segment(CumSum())(session, usd)
                  / Segment(CumSum())(session, usd / price))
trades["high"] = Segment(ExpandingMax())(session, price)
trades["low"] = Segment(ExpandingMin())(session, price)
last = trades.groupby(trades.index.floor("h")).last()   # display only
last[["price", "vwap", "high", "low"]].round(1)
```

```{code-cell} ipython3
fig, ax = plt.subplots(figsize=(9, 3.5))
ax.fill_between(trades.index, trades["low"], trades["high"], color="0.88",
                step="post", label="session high and low")
ax.plot(trades.index, trades["price"], color="0.3", lw=0.8, label="price")
ax.plot(trades.index, trades["vwap"], color="crimson", lw=1.2,
        label="session VWAP")
ax.set_ylabel("BTC price (USD)")
ax.legend(loc="upper left", fontsize=8)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
plt.tight_layout()
```

The data is six hours of Deribit BTC perpetual trades, with sizes in US
dollars, so each one-hour session stands in for a trading day. The label is the
hour number of each trade's timestamp. `Segment(op)(label, *inputs)` resets its
copy of `op` at every row whose label differs from the last label it saw, then
evaluates `op` on the row, and returns one output per input row. The VWAP
divides two running sums: the USD traded and the BTC traded, `usd / price`,
since the session began. The table holds the last value of each session,
grouped with pandas for display.

For daily sessions, use the day number as the label,
`timestamp // 86_400_000`; for futures, the contract's expiry as a number, so a
moving average restarts at each roll; for regimes, the state from a classifier.
A label that returns to an earlier value starts a new segment. With a `(T, K)`
input and one label per row, every column restarts on the same rows.

## Go deeper

- [`Segment`](../../functions_misc/Segment.md): the reset rule, `NaN` labels, and the match with pandas `groupby(...).transform(...)`.
- [`Resample`](../../functions_streams/Resample.md): one row per session instead of one row per input row.
- [Build bars from trades](../get-data-in/bars-from-ticks.md): fixed-time bars from the same trades.
