---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# Sample events at large cumulative moves

`CusumFilter` returns +1 when the cumulative rise since the last event passes
a threshold, -1 when the cumulative fall does, and 0 otherwise. The events
pick out the steps worth labelling or trading, more of them while the market
moves and fewer while it is quiet.

```{code-cell} ipython3
import pandas as pd
import matplotlib.pyplot as plt
from screamer import CusumFilter, LogReturn

df = pd.read_csv("../../notebooks/data/binance_daily_20.csv",
                 parse_dates=["date"])
df = df[df["symbol"] == "BTC"].set_index("date")

df["ret"] = LogReturn()(df["close"])
df["event"] = CusumFilter(threshold=0.1)(df["ret"])

events = df[df["event"] != 0]
events.groupby([events.index.year, "event"]).size().unstack()
```

The input is daily log returns, so `threshold=0.1` is a cumulative move of
about 10% since the last event. After an event the sum that fired restarts
from 0. The table counts the up (+1) and down (-1) events per year.

```{code-cell} ipython3
part = df.loc["2024":"2025"]
ups, downs = part[part["event"] > 0], part[part["event"] < 0]
fig, ax = plt.subplots(figsize=(9, 3.5))
ax.plot(part.index, part["close"], color="0.2", lw=1, label="close")
ax.scatter(ups.index, ups["close"], marker="^", color="seagreen", zorder=3,
           label="+1 (rise of 0.1)")
ax.scatter(downs.index, downs["close"], marker="v", color="crimson",
           zorder=3, label="-1 (fall of 0.1)")
ax.set_ylabel("BTC price (USD)")
ax.legend(loc="upper left", fontsize=8)
plt.tight_layout()
```

Events come close together while the price moves fast, as in the rise of
early 2024, and stop in quiet spells, such as December 2025. Down events
lead in the falling year 2022, 20 to 13, and up events in 2024, 18 to 10.

## Go deeper

- [`CusumFilter`](../../functions_signal/CusumFilter.md): the two sums, the reset rule, and `drift`.
- [`TripleBarrier`](../../functions_fin/TripleBarrier.md): label the outcome of a trade entered at each event.
- [`Hold`](../../functions_signal/Hold.md): keep a position for a fixed time after each event.
