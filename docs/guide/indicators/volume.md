---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# Volume indicators: OBV, MFI, accumulation/distribution and VWAP

`OBV` and `AD` add up volume signed by the price move, `MFI` is an RSI-like
ratio of up and down money flow, and `RollingVWAP` averages the price weighted
by volume.

```{code-cell} ipython3
import pandas as pd
import matplotlib.pyplot as plt
from screamer import OBV, AD, MFI, RollingVWAP

df = pd.read_csv("../../notebooks/data/binance_daily_20.csv",
                 parse_dates=["date"])
df = df[df["symbol"] == "BTC"].set_index("date")
h, l, c, v = df["high"], df["low"], df["close"], df["volume"]

df["obv"] = OBV()(c, v)          # volume signed by the close-to-close move
df["ad"] = AD()(h, l, c, v)      # volume weighted by the close in the range
df["mfi"] = MFI(14)(h, l, c, v)
df["vwap"] = RollingVWAP(20)(h, l, c, v)
print("change over 2022-2026, OBV:", df["obv"].iloc[-1] - df["obv"].iloc[0],
      "AD:", round(df["ad"].iloc[-1] - df["ad"].iloc[0]))
df[["close", "vwap", "obv", "ad", "mfi"]].tail().round(1)
```

```{code-cell} ipython3
part = df.loc["2025"]
fig, axes = plt.subplots(3, 1, figsize=(9, 6.5), sharex=True)
axes[0].plot(part.index, part["close"], color="0.6", lw=0.8, label="close")
axes[0].plot(part.index, part["vwap"], color="crimson", label="RollingVWAP(20)")
axes[0].set_ylabel("BTC (USD)")
axes[0].legend(loc="lower left", fontsize=8)
for col, color in [("obv", "steelblue"), ("ad", "darkorange")]:
    axes[1].plot(part.index, part[col] - part[col].iloc[0], color=color,
                 label=col.upper())
axes[1].set_ylabel("BTC volume\nsince 1 Jan")
axes[1].legend(loc="lower left", fontsize=8)
axes[2].plot(part.index, part["mfi"], color="seagreen")
for y in (20, 80):
    axes[2].axhline(y, color="0.5", ls="--", lw=0.8)
axes[2].set_ylabel("MFI(14)")
plt.tight_layout()
```

`OBV` and `AD` are running sums from the first bar, so their level depends on
where the data starts; read their changes, as in the middle panel, which counts
from 1 January 2025. The two disagree because they sign volume differently.
`OBV` counts the whole day's volume as buying when the close is above the
previous close. `AD` counts the share given by where the close sits between the
day's low and high. Over 2022 to 2026, `OBV` falls by 4.6 million BTC and `AD`
rises by 4.0 million.

`MFI` is an RSI-like ratio of up and down money flow: it sums the typical price
`(high + low + close) / 3` times volume over the up days and the down days of
the window, without Wilder's smoothing, so a move on heavy volume counts more;
80 and 20 are the usual overbought and oversold levels. `RollingVWAP` weights
the typical price of each of the last 20 days by its volume. For a VWAP that
restarts each session, sum price times volume and volume per session as on
[Restart a calculation each day, contract or
group](../portfolios/per-group.md).

## Go deeper

- [`OBV`](../../functions_rolling/OBV.md), [`AD`](../../functions_rolling/AD.md), [`MFI`](../../functions_rolling/MFI.md), [`RollingVWAP`](../../functions_rolling/RollingVWAP.md): formulas and TA-Lib equivalents.
- [`ADOSC`](../../functions_rolling/ADOSC.md): the Chaikin oscillator, fast minus slow average of the `AD` line.
- [Read order flow and liquidity](../order-flow/index.md): volume signed trade by trade instead of bar by bar.
