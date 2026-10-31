---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# Detect outliers robustly (median absolute deviation, sigma clipping)

Measure distance from the window median in units of `RollingMedianAD`, the
median absolute deviation, or clip with `RollingSigmaClip`, which leaves the
outliers out of its window; both keep a large outlier from widening the scale
that judges it.

```{code-cell} ipython3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from screamer import (LogReturn, RollingZscore, RollingMedian, RollingMad,
                      RollingMedianAD, RollingSigmaClip)

df = pd.read_csv("../../notebooks/data/binance_daily_20.csv",
                 parse_dates=["date"])
df = df[df["symbol"] == "BTC"].set_index("date")
r = df["ret"] = LogReturn()(df["close"])

df["z"] = RollingZscore(60)(r)
df["mean_ad"] = RollingMad(60)(r) * np.sqrt(np.pi / 2)  # mean absolute deviation
df["median_ad"] = RollingMedianAD(60)(r) * 1.4826       # median absolute deviation
df["z_robust"] = (r - RollingMedian(60)(r)) / df["median_ad"]
df["clipped"] = RollingSigmaClip(60, lower=-4, upper=4, output="nan")(r)

df["was_clipped"] = df["clipped"].isna() & df["z"].notna()
print("days with |z| > 5:", (df["z"].abs() > 5).sum(),
      "| robust |z| > 5:", (df["z_robust"].abs() > 5).sum(),
      "| clipped:", df["was_clipped"].sum())
df.loc["2022-11-08":"2022-11-10", ["ret", "z", "mean_ad", "median_ad", "z_robust"]].round(3)
```

```{code-cell} ipython3
flag = df[df["z_robust"].abs() > 5]
clip = df[df["was_clipped"]]
fig, ax = plt.subplots(figsize=(9, 3.5))
ax.plot(df.index, df["ret"], color="0.6", lw=0.6, label="BTC daily log return")
ax.scatter(flag.index, flag["ret"], s=40, facecolors="none",
           edgecolors="crimson", label="robust |z| > 5")
ax.scatter(clip.index, clip["ret"], s=12, marker="x", color="steelblue",
           label="RollingSigmaClip(60, -4, 4)")
ax.set_ylabel("log return")
ax.legend(loc="lower left", fontsize=8)
plt.tight_layout()
```

Each scale is multiplied to estimate the standard deviation of normal data:
by the square root of pi/2 for `RollingMad`, the mean absolute deviation, and
by 1.4826 for `RollingMedianAD`. On 9 November 2022, the day of the FTX
collapse, BTC had a log return of -0.15. The ordinary z-score includes that day
in its own standard deviation, so it reports only -4.5. The mean absolute
deviation grows on that day and the median absolute deviation does not, so the
robust z-score is -8.5. Over 2022 to 2026 the ordinary z-score passes 5 once
and the robust one 24 times.

`RollingSigmaClip` compares each return with the mean and standard deviation
of the window, leaves any return beyond 4 standard deviations out of the
window, and corrects the estimates for the clipping. With `output="nan"` it
returns `NaN` for those returns, 23 days here; the default
`output="clipped"` returns the bound instead.

## Go deeper

- [`RollingMedianAD`](../../functions_rolling/RollingMedianAD.md), [`RollingMad`](../../functions_rolling/RollingMad.md), [`RollingSigmaClip`](../../functions_rolling/RollingSigmaClip.md): definitions, warmup and the truncation correction.
- [Remove spikes from a series](despike.md): `Hampel` replaces outliers with the window median.
- [Separate jumps from normal volatility](../volatility/jumps.md): bipower variation, a scale that ignores isolated jumps.
