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
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from screamer import Resample, CombineLatest, Diff, RollingVarianceRatio

def closes(path):  # last trade price of each minute
    trades = pd.read_csv(path)
    return Resample(freq=60_000, agg="last", fill="carry")(
        trades["price"].to_numpy(), trades["timestamp"].to_numpy())

data = "../../notebooks/data/"
prices, ts = CombineLatest()(closes(data + "deribit_btc_perp_6h.csv"),
                             closes(data + "deribit_eth_perp_6h.csv"))
df = pd.DataFrame({"ratio": np.log(prices[:, 1] / prices[:, 0])},
                  index=pd.to_datetime(ts, unit="ms"))

df["change"] = Diff()(df["ratio"])
df["vr_z"] = RollingVarianceRatio(120, lag=5, output="zscore")(df["change"])
df.dropna().head().round(5)
```

The series is the log of the ETH/BTC price ratio, one value per minute over
six hours of Deribit perpetual trades. The test runs on its changes: over the
last 120 minutes, it compares the variance of 5-minute changes with 5 times
the variance of 1-minute changes. When minute changes tend to reverse, the
5-minute changes are smaller than that and the ratio is below 1. A z-score
below -1.96 rejects the random walk at the 5% level.

```{code-cell} ipython3
fig, (top, bottom) = plt.subplots(2, 1, figsize=(9, 5), sharex=True)
top.plot(df.index, df["ratio"], color="0.2", lw=0.8)
top.set_ylabel("log(ETH / BTC)")
bottom.plot(df.index, df["vr_z"], color="steelblue",
            label="RollingVarianceRatio(120, lag=5), z-score")
bottom.axhline(-1.96, color="crimson", ls="--", lw=0.8, label="-1.96")
bottom.set_ylabel("z-score")
bottom.legend(loc="lower right", fontsize=8)
bottom.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
plt.tight_layout()
```

From 02:00, when the first 120-minute window is full, the z-score stays below
-1.96 until 05:45, apart from two minutes at 05:05: at a one-minute scale the
ratio reverts. The z-score rises over the session as the reversion weakens.
On the daily closes of 2022 to 2026 in
`binance_daily_20.csv` the same test gives a variance ratio of 1.02 and a
z-score of 0.22, so across days the ratio behaves like a random walk. The
other pages of this chapter use the minute series.

## Go deeper

- [`RollingVarianceRatio`](../../functions_rolling/RollingVarianceRatio.md): the Lo-MacKinlay estimator and the robust z-statistic.
- [Estimate the Hurst exponent](hurst.md): a second test on the same series.
- [Pairs and mean reversion notebook](../../notebooks/18-pairs-and-mean-reversion.ipynb): hedge ratios and spreads between two assets.
