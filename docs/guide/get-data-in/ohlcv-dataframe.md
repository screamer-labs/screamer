---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# Compute indicators on an OHLCV DataFrame

Call each indicator on the DataFrame's columns and assign the result as new
columns, here `BollingerBands`, `RollingRSI`, and `ATR` on daily BTC bars.

```{code-cell} ipython3
import pandas as pd
import matplotlib.pyplot as plt
from screamer import BollingerBands, RollingRSI, ATR

df = pd.read_csv("../../notebooks/data/binance_daily_20.csv",
                 parse_dates=["date"])
df = df[df["symbol"] == "BTC"].set_index("date")

df[["bb_lower", "bb_mid", "bb_upper"]] = BollingerBands(20, 2.0)(df["close"])
df["rsi"] = RollingRSI(14)(df["close"])
df["atr"] = ATR(14)(df["high"], df["low"], df["close"])
df.iloc[20:25, 7:].round(1)
```

A function with one output returns one column. A function with several outputs
returns a `(T, k)` array that fills `k` columns in one assignment. `ATR` takes
three inputs in the order its page lists: high, low, close. Each indicator is
`NaN` until its window has filled.

```{code-cell} ipython3
year = df.loc["2025"]  # one year, to make the bands visible
fig, (ax0, ax1, ax2) = plt.subplots(3, 1, figsize=(9, 6), sharex=True,
                                    gridspec_kw={"height_ratios": [2, 1, 1]})
ax0.fill_between(year.index, year["bb_lower"], year["bb_upper"],
                 color="steelblue", alpha=0.15, label="BollingerBands(20, 2)")
ax0.plot(year.index, year["close"], color="0.2", lw=0.8, label="close")
ax0.set_ylabel("price (USD)")
ax0.legend(loc="upper left", fontsize=8)
ax1.plot(year.index, year["rsi"], color="darkorange", lw=0.8)
ax1.axhline(70, color="0.5", lw=0.6, ls="--")
ax1.axhline(30, color="0.5", lw=0.6, ls="--")
ax1.set_ylabel("RSI(14)")
ax2.plot(year.index, year["atr"], color="seagreen", lw=0.8)
ax2.set_ylabel("ATR(14)")
plt.tight_layout()
```

## Go deeper

- [Build bars from trades](bars-from-ticks.md): make the OHLCV DataFrame from a trade tape.
- [Financial indicators notebook](../../notebooks/03-financial-indicators.ipynb): more indicators on the same kind of data.
- [`BollingerBands`](../../functions_rolling/BollingerBands.md), [`RollingRSI`](../../functions_rolling/RollingRSI.md), [`ATR`](../../functions_rolling/ATR.md): parameters and column order.
- [NaN and warmup](../../nan_and_warmup.md): what each indicator returns before its window is full.
