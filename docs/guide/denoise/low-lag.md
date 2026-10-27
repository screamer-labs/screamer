---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# Smooth a series with less lag

`HullMA`, `KAMA` and `SuperSmoother` smooth a series and follow its turns
sooner than a `RollingMean` of the same length.

```{code-cell} ipython3
import pandas as pd
import matplotlib.pyplot as plt
from screamer import RollingMean, HullMA, KAMA, SuperSmoother

df = pd.read_csv("../../notebooks/data/binance_daily_20.csv",
                 parse_dates=["date"])
close = df[df["symbol"] == "BTC"].set_index("date")["close"]

out = pd.DataFrame({"close": close})
out["sma"] = RollingMean(30)(close)
out["hull"] = HullMA(30)(close)            # WMAs combined to cancel lag
out["kama"] = KAMA(30)(close)              # fast in trends, slow in chop
out["super"] = SuperSmoother(period=30)(close)  # 2-pole low-pass filter
out.loc["2025-03"].head().round(0)
```

```{code-cell} ipython3
part = out.loc["2025-01":"2025-06"]
fig, ax = plt.subplots(figsize=(9, 3.5))
ax.plot(part.index, part["close"], color="0.6", lw=0.8, label="close")
ax.plot(part.index, part["sma"], color="0.2", ls="--", label="RollingMean(30)")
ax.plot(part.index, part["hull"], color="crimson", label="HullMA(30)")
ax.plot(part.index, part["kama"], color="seagreen", label="KAMA(30)")
ax.plot(part.index, part["super"], color="steelblue",
        label="SuperSmoother(period=30)")
ax.set_ylabel("BTC price (USD)")
ax.legend(loc="lower left", fontsize=8)
plt.tight_layout()
```

`HullMA` subtracts a slow weighted average from twice a fast one, which
extrapolates the trend. `KAMA` changes its smoothing every step: it moves fast
when the last 30 days went mostly in one direction and slowly when they went
back and forth. `SuperSmoother` attenuates cycles shorter than `period` days.

In the late-April rally `HullMA` reaches the price within a week, while
`RollingMean(30)` is still about 8000 USD below it. `KAMA` hardly moves in
the first three weeks of April, while the price swings. The cost of less lag
is overshoot: after a sharp move `HullMA` runs past the price, as in
mid-February, when it keeps falling below the price after the drop at the start
of the month.

## Go deeper

- [`HullMA`](../../functions_rolling/HullMA.md), [`KAMA`](../../functions_rolling/KAMA.md), [`SuperSmoother`](../../functions_signal/SuperSmoother.md): formulas and warmup length.
- [Remove fast wiggles with a low-pass filter](lowpass.md): set the cutoff as a frequency.
- [Signal processing notebook](../../notebooks/04-signal-processing.ipynb): the trade-off between lag and smoothness.
