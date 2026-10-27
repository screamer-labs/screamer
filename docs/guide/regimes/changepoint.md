---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# Detect a change in mean or spread

`BayesianChangepoint` models a series as segments with their own mean and
spread and returns, at each step, the expected number of steps since the last
change. A drop of this run length towards 0 marks a change. Feed it a
standardized series, such as `RollingZscore` of returns.

```{code-cell} ipython3
import pandas as pd
import matplotlib.pyplot as plt
from screamer import BayesianChangepoint, LogReturn, RollingZscore

df = pd.read_csv("../../notebooks/data/binance_daily_20.csv",
                 parse_dates=["date"])
df = df[df["symbol"] == "BTC"].set_index("date")

df["ret"] = LogReturn()(df["close"])
df["z"] = RollingZscore(365)(df["ret"])
df["run_length"] = BayesianChangepoint(expected_run=250)(df["z"])[:, 0]

# Days where the run length falls from above 40 to below 15.
drop = (df["run_length"] < 15) & (df["run_length"].shift() > 40)
df.loc[drop, ["ret", "z", "run_length"]].round(3)
```

```{code-cell} ipython3
part = df.loc["2023":]
fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(9, 4.5), sharex=True)
ax0.plot(part.index, part["z"], color="0.5", lw=0.6)
ax0.set_ylabel("standardized return")
ax1.plot(part.index, part["run_length"], color="steelblue", lw=1)
marks = part[drop.loc["2023":]]
ax1.plot(marks.index, marks["run_length"], "o", color="crimson", ms=4,
         label="sharp drop")
ax1.set_ylabel("run length (days)")
ax1.legend(loc="upper left", fontsize=8)
plt.tight_layout()
```

`RollingZscore(365)` scales each return by the mean and standard deviation of
the last 365 days, so the values are of order 1, the scale the detector's
prior assumes. `expected_run=250` sets the prior chance of a change to 1 in
250 at each step. The run length grows by one a day while the segment holds;
the other two columns of the output are the mean and standard deviation of
the next value.

The table lists the five sharp drops, marked red in the plot. Each starts at
one large move, such as the log return of -0.076 on 17 August 2023 after a
quiet summer, and the log return of -0.151 on 5 February 2026, a standardized
return of -6.5. The mean of daily returns hardly changes, so the
changes found on this series are changes in spread. A slow fall of the run
length, as in mid-2025, means the evidence for a change builds up over weeks.

## Go deeper

- [`BayesianChangepoint`](../../functions_fin/BayesianChangepoint.md): the model, the prior, and `max_run`.
- [`RollingZscore`](../../functions_rolling/RollingZscore.md): the standardization.
- [Estimate the probability of a volatile regime](hmm.md): two fixed states instead of a change detector.
