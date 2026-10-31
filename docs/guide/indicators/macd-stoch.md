---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# Compute MACD and the stochastic oscillator (MACD, Stoch, ROC)

`MACD` gives the gap between a fast and a slow exponential average of the
close with its signal line, `Stoch` places the close within its recent
high-low range, and `ROC`, `ROCP` and `ROCR` give the change over `k` days as
a percentage, a fraction and a ratio.

```{code-cell} ipython3
import pandas as pd
import matplotlib.pyplot as plt
from screamer import MACD, Stoch, ROC, ROCP, ROCR

df = pd.read_csv("../../notebooks/data/binance_daily_20.csv",
                 parse_dates=["date"])
df = df[df["symbol"] == "BTC"].set_index("date")
h, l, c = df["high"], df["low"], df["close"]

df[["macd", "signal", "hist"]] = MACD(12, 26, 9)(c)   # TA-Lib MACD defaults
df[["k", "d"]] = Stoch(14, 3, 3)(h, l, c)             # talib.STOCH, slow form
df["roc"], df["rocp"], df["rocr"] = ROC(10)(c), ROCP(10)(c), ROCR(10)(c)

part = df.loc["2025"]
print("MACD crosses of the signal line in 2025:",
      int((part["hist"] > 0).astype(int).diff().abs().sum()))
print(f"Stoch %K above 80 on {(part['k'] > 80).mean():.0%} of 2025 days, "
      f"below 20 on {(part['k'] < 20).mean():.0%}")
df[["roc", "rocp", "rocr"]].iloc[-1].round(4)
```

```{code-cell} ipython3
fig, axes = plt.subplots(3, 1, figsize=(9, 6.5), sharex=True,
                         gridspec_kw={"height_ratios": [2, 1, 1]})
axes[0].plot(part.index, part["close"], color="0.2")
axes[0].set_ylabel("BTC close (USD)")
axes[1].bar(part.index, part["hist"], color="0.7", label="histogram")
axes[1].plot(part.index, part["macd"], color="steelblue", lw=0.9, label="MACD")
axes[1].plot(part.index, part["signal"], color="darkorange", lw=0.9,
             label="signal")
axes[1].legend(fontsize=8, loc="lower left")
axes[2].plot(part.index, part["k"], color="steelblue", lw=0.9, label="%K")
axes[2].plot(part.index, part["d"], color="darkorange", lw=0.9, label="%D")
for y in (20, 80):
    axes[2].axhline(y, color="0.5", ls="--", lw=0.8)
axes[2].set_ylabel("Stoch")
axes[2].legend(fontsize=8, loc="lower left")
plt.tight_layout()
```

The data are daily Binance BTC/USDT bars; the table shows the last day and the
plot shows 2025.

- `MACD(12, 26, 9)` returns three columns: the MACD line, the 12-day minus the
  26-day exponential average of the close (in USD); the signal line, a 9-day
  exponential average of the MACD line; and the histogram, their difference.
  The histogram changes sign when the MACD line crosses its signal line, 33
  times in 2025. The averages are `EwMean`, which starts at the first value,
  so the first weeks differ from TA-Lib's `MACD`, which starts from a simple
  average; the two converge after that.
- `Stoch(14, 3, 3)` takes the high, low and close and returns `%K` and `%D`.
  The raw value places the close within the 14-day high-low range, from 0 to
  100; `%K` is its 3-day simple average and `%D` the 3-day average of `%K`, as
  in TA-Lib's `STOCH` with these periods. `Stoch(14, 1, 3)` is the fast form,
  TA-Lib's `STOCHF`. Both lines are NaN for the first 17 days. In 2025 `%K`
  was above 80 on 22% of days and below 20 on 10%.
- `ROC(10)` is the 10-day change in percent, `ROCP(10)` the same change as a
  fraction (the same as `Return(10)`), and `ROCR(10)` the ratio of the close to
  the close 10 days earlier, as TA-Lib's functions of the same names. On the
  last day the close was 3.0% above its level ten days earlier: 3.0126,
  0.0301 and 1.0301.

## Go deeper

- [`MACD`](../../functions_rolling/MACD.md), [`Stoch`](../../functions_rolling/Stoch.md), [`ROC`](../../functions_fin/ROC.md): formulas, warmup and TA-Lib equivalents.
- [Compute momentum oscillators (Stochastic RSI, Williams %R, CCI, TRIX, Ultimate Oscillator)](oscillators.md): more readings of the same kind.
- [Conventions](../../conventions.md): why `MACD` differs from TA-Lib in the first samples.
- [Detect a trend with a moving-average crossover](../trend/crossover.md): the fast-minus-slow average as a trend signal.
