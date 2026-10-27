---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# Track a noisy level with a Kalman filter

`KalmanFilter` estimates a level that moves as a random walk from measurements
with noise, weighting each new measurement by how much it trusts it.

```{code-cell} ipython3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from screamer import KalmanFilter

trades = pd.read_csv("../../notebooks/data/deribit_btc_perp_6h.csv")
trades["time"] = pd.to_datetime(trades["timestamp"], unit="ms")
price = trades["price"]

kf = KalmanFilter(process_var=0.1,          # how far the level moves per trade
                  observation_var=4.0,      # noise of one trade price
                  initial_variance=np.inf)  # the first trade sets the level
trades["level"] = kf(price)
trades[["price", "level"]].tail().round(2)
```

The ratio `process_var / observation_var` sets the trade-off: a larger ratio
follows each trade more closely, a smaller one gives a smoother level that lags
more. Both variances are in the units of price squared, here USD squared.

```{code-cell} ipython3
part = trades.iloc[1000:1600]
fig, ax = plt.subplots(figsize=(9, 3.5))
ax.plot(part["time"], part["price"], ".", color="0.6", ms=3, label="trade price")
ax.plot(part["time"], part["level"], color="steelblue",
        label="KalmanFilter(process_var=0.1, observation_var=4)")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
ax.set_ylabel("BTC perpetual price (USD)")
ax.legend(loc="lower right", fontsize=8)
plt.tight_layout()
```

## Go deeper

- [`KalmanFilter`](../../functions_signal/KalmanFilter.md): the update equations and the `huber` option that limits the pull of outliers.
- [Filtering and forecasting with uncertainty notebook](../../notebooks/17-filtering-and-forecasting-with-uncertainty.ipynb): choosing the variances and reading the filter's uncertainty.
- [Remove spikes from a series](despike.md): clean bad prints before smoothing.
