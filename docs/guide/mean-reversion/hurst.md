---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# Estimate the Hurst exponent

`RollingHurst` estimates the Hurst exponent of a series of changes: near 0.5
when they are uncorrelated, above 0.5 when moves tend to continue, and below
0.5 when they tend to reverse.

```{code-cell} ipython3
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from screamer import Diff, RollingHurst

df = pd.read_csv("../../notebooks/data/deribit_ethbtc_ratio_1m.csv")
df.index = pd.to_datetime(df.pop("timestamp"), unit="ms")

df["change"] = Diff()(df["log_ratio"])
df["hurst"] = RollingHurst(120)(df["change"])
df["hurst"].describe().round(2)
```

```{code-cell} ipython3
fig, ax = plt.subplots(figsize=(9, 3.5))
ax.plot(df.index, df["hurst"], color="steelblue", label="RollingHurst(120)")
ax.axhline(0.5, color="0.5", ls="--", lw=0.8, label="0.5, a random walk")
ax.set_ylabel("Hurst exponent of log(ETH / BTC)")
ax.legend(loc="upper left", fontsize=8)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
plt.tight_layout()
```

The series is the minute log ETH/BTC ratio from the [variance ratio
page](variance-ratio.md). Like the variance ratio, the Hurst exponent is
estimated from the changes, not the level: on a price level it is close to 1
whatever the changes do. `RollingHurst(120)` splits the last 120 changes into
blocks of 4, 8, 16 and 32 and measures how the range of their running sum grows
with the block length. It gives `NaN` when a block has no variance, here when
the ratio did not change for a whole block.

The estimate stays between 0.29 and 0.45, below 0.5 throughout, which agrees
with the variance ratio: the ratio reverts.

## Go deeper

- [`RollingHurst`](../../functions_rolling/RollingHurst.md): the rescaled-range estimator and its small-sample correction.
- [Estimate how fast a series reverts](half-life.md): the next step on the same series.
