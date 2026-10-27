---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# Test whether a series reverts with the variance ratio

`RollingVarianceRatio(output="zscore")` tests the changes of a series against a
random walk: a variance ratio below 1 means the series reverts, above 1 that
it trends, and the z-score says whether the difference from 1 is significant.

```{code-cell} ipython3
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from screamer import Diff, RollingVarianceRatio

df = pd.read_csv("../../notebooks/data/deribit_ethbtc_ratio_1m.csv")
df.index = pd.to_datetime(df.pop("timestamp"), unit="ms")

df["change"] = Diff()(df["log_ratio"])
df["vr_z"] = RollingVarianceRatio(120, lag=5, output="zscore")(df["change"])
df.dropna().head().round(5)
```

```{code-cell} ipython3
fig, (top, bottom) = plt.subplots(2, 1, figsize=(9, 5), sharex=True)
top.plot(df.index, df["log_ratio"], color="0.2", lw=0.8)
top.set_ylabel("log(ETH / BTC)")
bottom.plot(df.index, df["vr_z"], color="steelblue",
            label="RollingVarianceRatio(120, lag=5), z-score")
bottom.axhline(-1.96, color="crimson", ls="--", lw=0.8, label="-1.96")
bottom.set_ylabel("z-score")
bottom.legend(loc="lower right", fontsize=8)
bottom.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
plt.tight_layout()
```

The series is the log of the ETH/BTC price ratio, one value per minute over six
hours of Deribit perpetual trades; `devtools/build_ethbtc_ratio.py` builds it
from the committed trade files with `Resample` and `CombineLatest`, taking the
last trade price of each asset in each minute. The test runs on its changes:
over the last 120 minutes, it compares the variance of 5-minute changes with 5
times the variance of 1-minute changes. When minute changes tend to reverse,
the 5-minute changes are smaller than that and the ratio is below 1. A z-score
below -1.96 rejects the random walk at the 5% level.

From 02:00, when the first 120-minute window is full, the z-score stays below
-1.96 until 05:45, apart from two minutes at 05:05: at a one-minute scale the
ratio reverts. The z-score rises over the session as the reversion weakens.
Six hours is one session, and part of the reversion at one minute is bid-ask
bounce: the last trade price jumps between the bid and the ask, and those
jumps reverse. On the daily ETH/BTC closes of 2022 to 2026 in
`binance_daily_20.csv`, one window over all 1733 daily changes with `lag=5`
gives a variance ratio of 1.02 and a z-score of 0.22, so across days the ratio
behaves like a random walk. The other pages of this chapter use the minute
series.

## Go deeper

- [`RollingVarianceRatio`](../../functions_rolling/RollingVarianceRatio.md): the Lo-MacKinlay estimator and the robust z-statistic.
- [Estimate the Hurst exponent](hurst.md): a second test on the same series.
- [Pairs and mean reversion notebook](../../notebooks/18-pairs-and-mean-reversion.ipynb): hedge ratios and spreads between two assets.
