---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# Estimate how fast a series reverts

`RollingOU` fits a mean-reverting Ornstein-Uhlenbeck model to a trailing
window and returns its mean-reversion rate per step, which converts to a
half-life: the number of steps in which a deviation from the mean halves.

```{code-cell} ipython3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from screamer import RollingOU

df = pd.read_csv("../../notebooks/data/deribit_ethbtc_ratio_1m.csv")
df.index = pd.to_datetime(df.pop("timestamp"), unit="ms")

df["mrr"] = RollingOU(120)(df["log_ratio"])
df["half_life"] = np.log(2) / -np.log(1 - df["mrr"])
df[["mrr", "half_life"]].dropna().iloc[::60].round(3)
```

```{code-cell} ipython3
fig, ax = plt.subplots(figsize=(9, 3.5))
ax.plot(df.index, df["half_life"], color="steelblue",
        label="half-life from RollingOU(120)")
ax.set_ylabel("half-life (minutes)")
ax.legend(loc="upper left", fontsize=8)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
plt.tight_layout()
```

The series is the minute log ETH/BTC ratio from the [variance ratio
page](variance-ratio.md). `RollingOU` fits the level, not the changes: each
step it regresses the ratio on its previous value over the last 120 minutes.
The mean-reversion rate `mrr` is the fraction of the distance to the mean
closed in one step, so the half-life is `ln 2 / -ln(1 - mrr)` steps, here
minutes. It holds for `mrr` between 0 and 1; outside that range the model does
not revert.

Until about 04:15 a deviation halves in one to four minutes. Then the
reversion slows, and from 05:00 the half-life is between 9 and 20 minutes.

## Go deeper

- [`RollingOU`](../../functions_rolling/RollingOU.md): the model, the fit, and the other outputs (`"mean"`, `"relmean"`, `"std"`).
- [`RollingADF`](../../functions_fin/RollingADF.md): a unit-root test on the level.
- [Trade the reversion with z-score bands](zscore-bands.md): the next step on the same series.
