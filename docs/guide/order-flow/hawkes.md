---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# Model clustered trade arrivals (Hawkes intensity)

`HawkesIntensity` gives the expected rate of events at each step from the
events before it: a baseline rate plus a bump from each recent event that
fades geometrically, so the rate is high just after a burst of trading.

```{code-cell} ipython3
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from screamer import FillNa, HawkesIntensity, Resample, Sign

trades = pd.read_csv("../../notebooks/data/deribit_btc_perp_6h.csv")
# Trades per second, 0 for a second without trades.
counts, index = Resample(freq=1_000, agg="count", fill="nan")(
    trades["price"].to_numpy(), trades["timestamp"].to_numpy())
active = Sign()(FillNa(0.0)(counts))    # 1 for a second with a trade, else 0

intensity = HawkesIntensity(decay=0.9, alpha=0.05, mu=0.025)(active)
sec = pd.DataFrame({"active": active, "intensity": intensity},
                   index=pd.to_datetime(index, unit="ms"))
quintile = pd.qcut(sec["intensity"], 5, labels=[1, 2, 3, 4, 5])
sec.groupby(quintile, observed=True).mean().round(3)
```

```{code-cell} ipython3
w = sec.loc["2023-08-04 02:20":"2023-08-04 02:30"]
fig, ax = plt.subplots(figsize=(9, 3.5))
ax.vlines(w.index[w["active"] > 0], 0, 1, color="0.7", lw=0.8,
          label="second with a trade")
ax.plot(w.index, w["intensity"], color="crimson", lw=1.2,
        label="HawkesIntensity")
ax.set_ylabel("events per second")
ax.legend(loc="upper left", fontsize=8)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
plt.tight_layout()
```

The trades are six hours of Deribit BTC perpetual trades.
`Resample(freq=1_000, agg="count", fill="nan")` counts the trades in each
second and gives `NaN` for a second without trades, which `FillNa` sets to 0. Many trades share a
timestamp, the pieces of one order that fills against several resting orders,
so the events here are seconds with at least one trade, which `Sign` turns
into 1. A trade happened in 5% of the seconds.

The intensity is `mu + kappa`: each event adds `alpha` to `kappa`, and `kappa`
keeps the fraction `decay` of itself from one second to the next. It is
computed before the second's own event enters, so each value is a forecast for
that second from the seconds before. The three parameters are set by hand
here, not fitted: a rate of 0.025 events per second when nothing has happened,
0.05 added per event and then 90% kept per second, so the next second sees a
bump of 0.045, which halves about every 7 seconds.

The table groups the seconds into fifths by intensity. In the top fifth, 11.7%
of the seconds held a trade, close to its mean intensity of 0.10, against 3.3
to 4.3% in the other four: trades come in clusters, and the intensity finds
them from the past alone.

## Go deeper

- [`HawkesIntensity`](../../functions_micro/HawkesIntensity.md): the recursion and the choice of marks.
- [`Resample`](../../functions_streams/Resample.md): counts per interval and the `fill` options for empty intervals.
- [Measure order-flow toxicity (VPIN)](vpin.md): how one-sided the clustered trades are.
