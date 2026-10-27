---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# Measure the average true range

`ATR` averages the true range of each bar, the largest of high minus low and
the distances from the previous close to the high and the low, with Wilder's
smoothing.

```{code-cell} ipython3
import pandas as pd
import matplotlib.pyplot as plt
from screamer import ATR

df = pd.read_csv("../../notebooks/data/binance_daily_20.csv",
                 parse_dates=["date"])
df = df[df["symbol"] == "BTC"].set_index("date")

df["atr"] = ATR(14)(df["high"], df["low"], df["close"])
df["atr_pct"] = df["atr"] / df["close"] * 100
df[["high", "low", "close", "atr", "atr_pct"]].tail().round(1)
```

```{code-cell} ipython3
fig, ax = plt.subplots(figsize=(9, 3.5))
ax.plot(df.index, df["atr"], color="seagreen", label="ATR(14)")
ax.set_ylabel("BTC ATR (USD)")
ax.legend(loc="upper left", fontsize=8)
plt.tight_layout()
```

The inputs come in the order high, low, close. `ATR` is in price units, here
USD, so it suits stop distances and position sizes in the same units.
`ATR(14)` is `NaN` for the first 14 bars and matches `talib.ATR` after that.

The ATR grows with the price level as well as with volatility. Dividing by the
close, as in `atr_pct`, gives a measure that compares across price levels;
`NATR` computes it directly.

## Go deeper

- [`ATR`](../../functions_rolling/ATR.md), [`TrueRange`](../../functions_rolling/TrueRange.md), [`NATR`](../../functions_rolling/NATR.md): the true range, its Wilder average, and the average as a percentage of the close.
- [Size a position by volatility](../signal-to-position/volatility-sizing.md): use ATR or the standard deviation of returns to set position sizes.
