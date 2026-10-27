---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# Build a feature matrix for a model

A feature matrix is one causal screamer call per column, stacked with
`np.column_stack`: here `LogReturn`, `RollingZscore`, and `EwMean` of `OFI`,
computed on 10-second bars from `Resample`.

```{code-cell} ipython3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from screamer import Resample, LogReturn, RollingZscore, EwMean, OFI

trades = pd.read_csv("../../notebooks/data/deribit_btc_perp_6h.csv")
ticks = np.column_stack([trades["price"], trades["volume"]])
bars, ts = Resample(freq=10_000, agg="ohlcv2", fill="carry")(
    ticks, trades["timestamp"].to_numpy())
close, buy, sell = bars[:, 3], bars[:, 4], bars[:, 5]

X = np.column_stack([
    LogReturn(1)(close),               # last bar's return
    LogReturn(6)(close),               # return over the last minute
    RollingZscore(60)(close),          # close against its last ten minutes
    EwMean(span=6)(OFI()(buy, sell)),  # recent order-flow imbalance
])
names = ["return 10s", "return 1m", "z-score 10m", "flow"]
print(X.shape, "rows with a NaN:", np.isnan(X).any(axis=1).sum())
```

The trades are six hours of the Deribit BTC perpetual, with the timestamp in
milliseconds and the volume signed by the aggressor side. `agg="ohlcv2"`
returns each 10-second bar's open, high, low, and close, then its buy and sell
volume in columns 4 and 5, which `OFI` turns into an imbalance from -1 to 1.
Row `t` of `X` uses bars up to `t` only. The first 59 rows hold a NaN while
the 60-bar z-score fills.

```{code-cell} ipython3
hours = (ts - ts[0]) / 3.6e6
fig, axes = plt.subplots(len(names), 1, figsize=(9, 6), sharex=True)
for ax, column, name in zip(axes, X.T, names):
    ax.plot(hours, column, lw=0.6)
    ax.set_ylabel(name, fontsize=8)
axes[-1].set_xlabel("hours")
plt.tight_layout()
```

## Go deeper

- [Pair features with labels without look-ahead](training-set.md): turn `X` and a target into training rows.
- [Supervised forecasting](../../notebooks/16-supervised-forecasting.ipynb): fit a model on these features, use time-based horizons, and backtest the prediction.
- [Build bars from ticks](../get-data-in/bars-from-ticks.md): the `Resample` bar types.
- [`OFI`](../../functions_micro/OFI.md), [`RollingZscore`](../../functions_rolling/RollingZscore.md), [`Resample`](../../functions_streams/Resample.md): the functions used here.
