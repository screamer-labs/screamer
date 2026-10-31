---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# Momentum oscillators: Stochastic RSI, Williams %R, CCI, TRIX, Ultimate Oscillator

`StochRSI`, `WilliamsR`, `CCI`, `TRIX` and `UltimateOscillator` each turn
recent prices into one momentum reading, and each takes the inputs and
periods of its TA-Lib counterpart.

```{code-cell} ipython3
import pandas as pd
import matplotlib.pyplot as plt
from screamer import StochRSI, WilliamsR, CCI, TRIX, UltimateOscillator

df = pd.read_csv("../../notebooks/data/binance_daily_20.csv",
                 parse_dates=["date"])
df = df[df["symbol"] == "BTC"].set_index("date")
h, l, c = df["high"], df["low"], df["close"]

df["stoch_rsi"] = StochRSI(14, 14, 3, 3)(c)[:, 0]   # %K, the slow form
df["williams_r"] = WilliamsR(14)(h, l, c)
df["cci"] = CCI(20)(h, l, c)
df["trix"] = TRIX(15)(c)
df["ultimate"] = UltimateOscillator(7, 14, 28)(h, l, c)
cols = ["stoch_rsi", "williams_r", "cci", "trix", "ultimate"]
print("CCI within +-100:", f"{(df['cci'].abs() <= 100).mean():.0%}",
      "| TRIX sign changes in 2025:",
      int((df["trix"] > 0).astype(int).diff().abs().loc["2025"].sum()))
df[cols].describe().loc[["min", "max"]].round(2)
```

```{code-cell} ipython3
part = df.loc["2025"]
levels = {"stoch_rsi": (20, 80), "williams_r": (-80, -20), "cci": (-100, 100),
          "trix": (0,), "ultimate": (30, 70)}
fig, axes = plt.subplots(6, 1, figsize=(9, 9), sharex=True)
axes[0].plot(part.index, part["close"], color="0.2")
axes[0].set_ylabel("BTC close")
for ax, col in zip(axes[1:], cols):
    ax.plot(part.index, part[col], color="steelblue", lw=0.9)
    for y in levels[col]:
        ax.axhline(y, color="0.5", ls="--", lw=0.8)
    ax.set_ylabel(col, fontsize=8)
plt.tight_layout()
```

The dashed lines are the usual overbought and oversold levels, and zero for
`TRIX`.

- `StochRSI` places the RSI within its own 14-day range, from 0 to 100.
  `StochRSI(14, 14, 3, 3)` smooths `%K` over 3 days; `smooth_k=1` gives the
  fast form of TA-Lib's `STOCHRSI`. The second output column is `%D`.
- `WilliamsR` places the close within the 14-day high-low range, from -100
  (at the low) to 0 (at the high).
- `CCI` is the distance of the typical price from its 20-day mean, in units of
  0.015 times its mean absolute deviation. It has no bounds; on these BTC days
  it is within ±100 about 60% of the time.
- `TRIX` is the one-day percentage change of a triple-smoothed exponential
  average. It is the slowest of the five and changes sign 11 times in 2025.
- `UltimateOscillator` weights buying pressure over 7, 14 and 28 days by 4, 2
  and 1. Averaging three periods keeps it in a narrower band than the others:
  between 23 and 78 on these data.

`WilliamsR`, `CCI` and `UltimateOscillator` move together: over 2022 to 2026
the correlation of `WilliamsR` with each of the other two is near 0.8 or
above, so using several of them adds little.

## Go deeper

- [`StochRSI`](../../functions_rolling/StochRSI.md), [`WilliamsR`](../../functions_rolling/WilliamsR.md), [`CCI`](../../functions_rolling/CCI.md), [`TRIX`](../../functions_rolling/TRIX.md), [`UltimateOscillator`](../../functions_rolling/UltimateOscillator.md): formulas, warmup, and TA-Lib equivalents.
- [`Stoch`](../../functions_rolling/Stoch.md): the stochastic oscillator on price.
- [Trade RSI reversion](../strategies/rsi-reversion.md): an oscillator turned into a strategy.
