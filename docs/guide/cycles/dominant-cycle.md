---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# Measure the dominant cycle length

`DominantCycle` measures the period of the strongest cycle in a series, in
samples, with John Ehlers' Hilbert transform method. Remove the trend and the
fastest noise first with `RoofingFilter`, a bandpass, because a trend biases
the measurement.

```{code-cell} ipython3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from screamer import RoofingFilter, DominantCycle

df = pd.read_csv("../../notebooks/data/binance_daily_20.csv",
                 parse_dates=["date"])
df = df[df["symbol"] == "BTC"].set_index("date")

def cycle_length(x):
    band = RoofingFilter(hp_period=48, lp_period=10)(x)
    return DominantCycle()(band)

df["period"] = cycle_length(np.log(df["close"]))
# The same measurement on a random walk of the same length.
walk = np.cumsum(np.random.default_rng(0).standard_normal(len(df)))
df["walk_period"] = cycle_length(walk)
df.loc["2023":, ["period", "walk_period"]].describe().round(1)
```

```{code-cell} ipython3
part = df.loc["2023":]
fig, ax = plt.subplots(figsize=(9, 3.5))
ax.plot(part.index, part["period"], color="steelblue", lw=1,
        label="BTC log close")
ax.plot(part.index, part["walk_period"], color="0.6", lw=1,
        label="random walk")
ax.set_ylabel("dominant cycle (days)")
ax.legend(loc="upper left", fontsize=8)
plt.tight_layout()
```

`RoofingFilter(hp_period=48, lp_period=10)` keeps cycles between 10 and 48
days. Its state starts at zero, so on a price level its first months carry a
large start-up swing; the table and the plot start in 2023. `DominantCycle`
reports a period between 6 and 50 samples, here days.

The BTC period moves between about 14 and 42 days with a median near 20. A
random walk through the same filter reads about the same, so this measurement
alone does not show that BTC has a cycle: the bandpass sets the range of the
answer. The period is useful as a setting for tools that need a cycle length,
such as [`CycleSine`](turning-points.md).

## Go deeper

- [`DominantCycle`](../../functions_signal/DominantCycle.md): the method, the 6 to 50 sample range, and the bias from a trend.
- [`RoofingFilter`](../../functions_signal/RoofingFilter.md): the two cutoffs of the bandpass.
- [Cycle and spectral analysis notebook](../../notebooks/19-cycle-and-spectral-analysis.ipynb): phase, frequency, and amplitude on test signals with a known cycle.
