---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# Keltner Channels next to Bollinger Bands

`KeltnerChannels` sets its bands a multiple of the average true range from an
exponential average, while `BollingerBands` sets them a multiple of the
standard deviation of the close from a simple average.

```{code-cell} ipython3
import pandas as pd
import matplotlib.pyplot as plt
from screamer import KeltnerChannels, BollingerBands

df = pd.read_csv("../../notebooks/data/binance_daily_20.csv",
                 parse_dates=["date"])
df = df[df["symbol"] == "BTC"].set_index("date")

kc = KeltnerChannels(20, 1.5)(df["high"], df["low"], df["close"])
bb = BollingerBands(20, 2.0)(df["close"])
df["kc_lower"], df["kc_upper"] = kc[:, 0], kc[:, 2]
df["bb_lower"], df["bb_upper"] = bb[:, 0], bb[:, 2]
df["squeeze"] = ((df["bb_lower"] > df["kc_lower"])
                 & (df["bb_upper"] < df["kc_upper"]))

width = pd.DataFrame({
    "Keltner": (df["kc_upper"] - df["kc_lower"]) / kc[:, 1],
    "Bollinger": (df["bb_upper"] - df["bb_lower"]) / bb[:, 1]})
width.describe().loc[["min", "50%", "max"]].round(3)
```

```{code-cell} ipython3
part = df.loc["2025"]
fig, ax = plt.subplots(figsize=(9, 3.5))
ax.plot(part.index, part["close"], color="0.2", lw=1, label="close")
ax.plot(part.index, part[["kc_lower", "kc_upper"]], color="darkorange",
        label=["KeltnerChannels(20, 1.5)", None])
ax.plot(part.index, part[["bb_lower", "bb_upper"]], color="steelblue",
        ls="--", label=["BollingerBands(20, 2.0)", None])
ax.fill_between(part.index, *ax.get_ylim(), where=part["squeeze"],
                color="0.85", label="Bollinger inside Keltner")
ax.set_ylabel("BTC price (USD)")
ax.legend(loc="upper left", fontsize=8)
plt.tight_layout()
```

The table gives the width of each channel as a share of its middle line. The
Keltner width follows the average true range, which includes each day's high
and low, so it changes slowly: from 6% to 24% of the price over 2022 to 2026.
The Bollinger width follows the standard deviation of 20 closes. It narrows to
2.6% in a quiet stretch and widens to 74% after a crash, when the crash's
returns stay in the window for 20 days.

A day on which the Bollinger Bands lie inside the Keltner Channels, shaded in
the plot, marks closes that vary less than the daily ranges suggest. Traders
call this a squeeze and read it as a calm before a larger move; the pairing of
1.5 ATR with 2 standard deviations is John Carter's. On these data it holds on
18% of the days.

## Go deeper

- [`KeltnerChannels`](../../functions_rolling/KeltnerChannels.md), [`BollingerBands`](../../functions_rolling/BollingerBands.md): the formulas and the warmup.
- [Measure the average true range](../volatility/atr.md): the range behind the Keltner width.
- [Fade moves outside the Bollinger Bands](../strategies/bollinger-reversion.md): the bands in a strategy.
