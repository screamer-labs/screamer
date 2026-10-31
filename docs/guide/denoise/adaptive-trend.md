---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# Track an adaptive trend (MAMA, instantaneous trendline)

`MAMA` and `InstantaneousTrendline` are John Ehlers' smoothers that set their
own amount of smoothing from the cycle they measure in the series, so they
need no window length.

```{code-cell} ipython3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from screamer import MAMA, InstantaneousTrendline

df = pd.read_csv("../../notebooks/data/binance_daily_20.csv",
                 parse_dates=["date"])
df = df[df["symbol"] == "BTC"].set_index("date")

pair = MAMA(fast_limit=0.5, slow_limit=0.05)(df["close"])
df["mama"], df["fama"] = pair[:, 0], pair[:, 1]
df["trendline"] = InstantaneousTrendline()(df["close"])

side = np.sign(df["mama"] - df["fama"])      # +1 while MAMA is above FAMA
print("MAMA-FAMA crossings in 2025:", int((side.diff().abs() > 0).loc["2025"].sum()))
df[["close", "mama", "fama", "trendline"]].tail().round(0)
```

```{code-cell} ipython3
part = df.loc["2025-01":"2025-06"]
fig, ax = plt.subplots(figsize=(9, 3.5))
ax.plot(part.index, part["close"], color="0.6", lw=0.8, label="close")
ax.plot(part.index, part["mama"], color="crimson", label="MAMA")
ax.plot(part.index, part["fama"], color="steelblue", label="FAMA")
ax.plot(part.index, part["trendline"], color="seagreen",
        label="InstantaneousTrendline")
ax.set_ylabel("BTC price (USD)")
ax.legend(loc="lower left", fontsize=8)
plt.tight_layout()
```

`MAMA` is an exponential average whose smoothing constant changes every day
with the rate at which the phase of the dominant cycle advances. While a cycle
rotates steadily, `MAMA` smooths heavily and stays flat. Twice per cycle the
measured phase wraps around, the constant jumps to `fast_limit`, and `MAMA`
moves most of the way to the price in one step. This is why it moves in
stairs. `FAMA` applies half the constant to `MAMA` and lags it, and Ehlers
reads a crossing of the two as a change of trend; on BTC there were 18 in
2025.

`InstantaneousTrendline` averages the price over one measured dominant cycle,
which removes that cycle and keeps the trend. It follows the price smoothly,
about half a cycle behind. All three outputs are `NaN` for the first 40 days.

## Go deeper

- [`MAMA`](../../functions_signal/MAMA.md), [`InstantaneousTrendline`](../../functions_signal/InstantaneousTrendline.md): the formulas and how they match TA-Lib's `MAMA` and `HT_TRENDLINE`.
- [Smooth a series with less lag](low-lag.md): `KAMA`, another adaptive average, set by a window length.
- [Measure the dominant cycle length](../cycles/dominant-cycle.md): the cycle measurement these smoothers rely on.
