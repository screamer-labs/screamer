---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# Make a series stationary without losing its memory (fractional differencing)

`FracDiff(d)` differences a series by a fractional order `d` between 0 and 1:
the smallest `d` at which `RollingADF` rejects a unit root gives a stationary
feature that still tracks the price level, which a first difference does not.

```{code-cell} ipython3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from screamer import FracDiff, RollingADF, Log

df = pd.read_csv("../../notebooks/data/binance_daily_20.csv")
x = Log()(df.loc[df["symbol"] == "BTC", "close"].to_numpy())

orders, adf, corr = np.round(np.arange(0, 1.05, 0.1), 1), [], []
for d in orders:
    y = FracDiff(d=d, window_size=250, threshold=0.0)(x)
    adf.append(RollingADF(window_size=1000)(y)[-1])
    ok = ~np.isnan(y)
    corr.append(np.corrcoef(x[ok], y[ok])[0, 1])
d_min = orders[np.argmax(np.array(adf) < -2.86)]
print(f"smallest d with ADF below -2.86: {d_min}")
print(pd.DataFrame({"d": orders, "ADF": adf, "corr": corr}).round(2).T)
```

```{code-cell} ipython3
fig, ax = plt.subplots(figsize=(8, 3.5))
ax.plot(orders, adf, "o-", color="crimson", label="ADF statistic")
ax.axhline(-2.86, color="crimson", ls="--", lw=0.8,
           label="5% critical value")
ax.set_xlabel("differencing order d")
ax.set_ylabel("ADF statistic")
ax2 = ax.twinx()
ax2.plot(orders, corr, "s-", color="steelblue",
         label="correlation with log price")
ax2.set_ylabel("correlation with log price")
fig.legend(loc="lower left", bbox_to_anchor=(0.1, 0.2), fontsize=8)
plt.tight_layout()
```

The series is the log close of daily BTC from January 2022 to September 2026.
For each order `d` from 0 to 1, `FracDiff` applies 250 weights, so each value
uses the last 250 days, and `RollingADF` gives the augmented Dickey-Fuller
statistic over the last 1000 days. A statistic below the 5% critical value of
-2.86 rejects a unit root. `d = 0` is the log price itself and `d = 1` the
daily log return.

The log price, at -2.34, does not pass. At `d = 0.2` the statistic is -3.81
and the series still has a correlation of 0.97 with the log price; the daily
return passes easily but has none. A model that takes the `d = 0.2` series as a
feature sees a stationary input that keeps most of the information in the price
level. The first 249 values are NaN while the window fills, and every value
uses past prices only.

## Go deeper

- [Build a feature matrix for a model](features.md): stack features like this one into columns.
- [`FracDiff`](../../functions_signal/FracDiff.md): the weights, truncation with `threshold`, and the warmup policies.
- [`RollingADF`](../../functions_fin/RollingADF.md): the regression behind the statistic, and its critical values.
