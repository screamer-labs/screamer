---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# Smooth a series with a moving average

`RollingMean` averages the last `window_size` values with equal weight, and
`EwMean` averages all past values with weights that decay geometrically, so
recent values count more.

```{code-cell} ipython3
import pandas as pd
import matplotlib.pyplot as plt
from screamer import RollingMean, EwMean

df = pd.read_csv("../../notebooks/data/binance_daily_20.csv",
                 parse_dates=["date"])
df = df[df["symbol"] == "BTC"].set_index("date")

df["sma"] = RollingMean(50)(df["close"])
df["ema"] = EwMean(span=50)(df["close"])
df[["close", "sma", "ema"]].iloc[48:53].round(0)
```

```{code-cell} ipython3
year = df.loc["2024"]
fig, ax = plt.subplots(figsize=(9, 3.5))
ax.plot(year.index, year["close"], color="0.6", lw=0.8, label="close")
ax.plot(year.index, year["sma"], color="steelblue", label="RollingMean(50)")
ax.plot(year.index, year["ema"], color="darkorange", label="EwMean(span=50)")
ax.set_ylabel("BTC price (USD)")
ax.legend(loc="upper left", fontsize=8)
plt.tight_layout()
```

`RollingMean(50)` is `NaN` for the first 49 days, while its window is filling.
`EwMean` returns a value from the first day and gives the same result as pandas
`ewm(span=50).mean()`.

Both averages trail the price. `EwMean` weights the latest days more, so it
turns sooner after a change in direction.

## Go deeper

- [Smooth a series with less lag](low-lag.md): averages that follow the price more closely.
- [`RollingMean`](../../functions_rolling/RollingMean.md) and [`EwMean`](../../functions_ew/EwMean.md): parameters and the four ways to set the decay of `EwMean`.
- [Signal processing notebook](../../notebooks/04-signal-processing.ipynb): moving averages next to other filters.
