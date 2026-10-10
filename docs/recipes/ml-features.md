---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# A feature matrix for machine learning

screamer computes features as causal columns, and `forecast_pairs` aligns each
row of features with the target value from a fixed horizon later. The result is
a training set for any batch learner. This recipe builds four features from
10-second BTC bars, pairs them with the next minute's return, and fits a linear
model with NumPy.

## Features

Each feature is a screamer call on the bar data, stacked into a `(T, 4)` matrix.
`agg="ohlcv2"` splits each bar's volume into buyer- and seller-initiated parts
(columns 4 and 5), which gives an order-flow imbalance between -1 and 1.

```{code-cell} ipython3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from screamer import Resample, LogReturn, RollingZscore, EwMean, forecast_pairs

trades = pd.read_csv("../notebooks/data/deribit_btc_perp_6h.csv")
bars, ts = Resample(freq=10_000, agg="ohlcv2", fill="carry")(
    np.column_stack([trades["price"], trades["volume"]]), trades["timestamp"].to_numpy())
close, buy, sell = bars[:, 3], bars[:, 4], bars[:, 5]

imbalance = (buy - sell) / np.maximum(buy + sell, 1.0)
X = np.column_stack([
    LogReturn(1)(close),            # last bar's return
    LogReturn(6)(close),            # return over the last minute
    RollingZscore(60)(close),       # price against its last ten minutes
    EwMean(span=6)(imbalance),      # recent order-flow imbalance
])
names = ["ret 10s", "ret 1m", "zscore 10m", "flow"]
X.shape
```

## Target and pairs

The target is the trailing one-minute return, `LogReturn(6)(close)`: at row `t`
it is the return from `t - 6` to `t`. `forecast_pairs(X, y, count=6)` delays the
features by six bars, so each row pairs the features known at `t - 6` with the
return over the six bars that followed. `dropna=True` drops the warmup rows.

```{code-cell} ipython3
y = LogReturn(6)(close)
Xs, ys = forecast_pairs(X, y, count=6, dropna=True)
print(f"{len(Xs)} rows of {Xs.shape[1]} features")
```

## Fit and evaluate

The rows are in time order, so the first 60% train the model and the rest test
it. The model is ordinary least squares on standardized features; any
scikit-learn style learner takes the same `Xs` and `ys`.

```{code-cell} ipython3
cut = int(0.6 * len(Xs))
mu, sd = Xs[:cut].mean(axis=0), Xs[:cut].std(axis=0)
A = np.column_stack([np.ones(len(Xs)), (Xs - mu) / sd])
coef, *_ = np.linalg.lstsq(A[:cut], ys[:cut], rcond=None)
pred = A[cut:] @ coef
corr = np.corrcoef(pred, ys[cut:])[0, 1]
print(f"out-of-sample correlation of prediction and realized return: {corr:+.3f}")
```

The test set covers about two hours of overlapping one-minute returns, so the
correlation is a noisy estimate; it illustrates the workflow rather than an
edge. The left panel compares the out-of-sample prediction with the realized
one-minute return; the right panel shows the fitted weight of each standardized
feature.

```{code-cell} ipython3
fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(10, 3.8),
                               gridspec_kw={"width_ratios": [3, 2]})
ax0.scatter(pred * 1e4, ys[cut:] * 1e4, s=4, alpha=0.3)
ax0.axhline(0, color="k", lw=0.4)
ax0.axvline(0, color="k", lw=0.4)
ax0.set_xlabel("predicted return (bps)")
ax0.set_ylabel("realized return (bps)")
ax1.barh(names, coef[1:] * 1e4, color="steelblue")
ax1.axvline(0, color="k", lw=0.5)
ax1.set_xlabel("weight (bps per std)")
plt.tight_layout()
```

The [supervised forecasting notebook](../notebooks/16-supervised-forecasting.ipynb)
covers time-based horizons with `duration=` and turns a prediction into a
backtest.

Related pages: [`forecast_pairs`](../functions_supervised/forecast_pairs.md),
[`LogReturn`](../functions_fin/LogReturn.md),
[`RollingZscore`](../functions_rolling/RollingZscore.md), and
[`Resample`](../functions_streams/Resample.md).
