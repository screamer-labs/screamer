---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# Pair features with labels without look-ahead

`forecast_pairs(X, y, count=h)` builds a training set: row `t` holds the
features `X[t]` and the label of the `h` bars after `t`.

```{code-cell} ipython3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from screamer import (LogReturn, RollingZscore, RollingStd, TripleBarrierOHLC,
                      forecast_pairs)

df = pd.read_csv("../../notebooks/data/binance_daily_20.csv",
                 parse_dates=["date"])
btc = df[df["symbol"] == "BTC"].set_index("date")
o, h, l, c = (btc[k].to_numpy() for k in ["open", "high", "low", "close"])

X = np.column_stack([LogReturn(20)(c), RollingZscore(50)(c),
                     RollingStd(20)(LogReturn(1)(c))])
horizon = 10
y = TripleBarrierOHLC(horizon=horizon, upper=1.0, lower=1.0,
                      barrier_unit="vol", vol_span=20)(o, h, l, c)[:, 0]

Xp, yp = forecast_pairs(X, y, count=horizon)
print(pd.DataFrame(np.column_stack([Xp, yp]), index=btc.index).tail(12).round(3))
Xs, ys = forecast_pairs(X, y, count=horizon, dropna=True)
print(f"{len(Xs)} training rows of {Xs.shape[1]} features")
```

Columns 0 to 2 are the features: the 20-day log return, the z-score of the
close over the last 50 days, and the standard deviation of the last 20 daily returns, all
known at the close of day `t`. Column 3 is the triple-barrier label of
[Label each bar with its future move](labels.md), which `TripleBarrierOHLC`
stamps 10 days after its bar and `forecast_pairs` moves back to the row of
that bar. The last 10 rows have no label yet: they are the rows a live model
predicts. `dropna=True` keeps the 1675 rows where every feature and the label
are known, ready for any scikit-learn style learner. The rows stay in time
order, so the first part can train a model and the rest test it. Leave
`horizon` rows out between the two parts, because the labels of the last
training rows cover days inside the test part.

```{code-cell} ipython3
last = slice(len(y) - 40, len(y))
fig, ax = plt.subplots(figsize=(9, 3))
ax.scatter(btc.index[last], y[last] + 0.08, s=14, color="0.6",
           label="TripleBarrierOHLC output, stamped when known")
ax.scatter(btc.index[last], yp[last] - 0.08, s=14, color="crimson",
           label="after forecast_pairs, at the labeled bar")
ax.set_yticks([-1, 0, 1])
ax.set_ylabel("label")
ax.legend(fontsize=8, loc="lower left")
plt.tight_layout()
```

Over the last 40 days the red points repeat the grey ones 10 days earlier, and
end 10 days before the data does.

## Go deeper

- [Build a feature matrix for a model](features.md): features from tick data.
- [Labels and training sets](../../labels.md): pairing return, volatility, range, and rank targets, and rows across many assets.
- [Supervised forecasting](../../notebooks/16-supervised-forecasting.ipynb): fit a model, use time-based horizons with `duration=`, and backtest the prediction.
- [Classifying direction](../../notebooks/23-classifying-direction.ipynb): train a classifier on direction labels.
- [`forecast_pairs`](../../functions_supervised/forecast_pairs.md): `count=` and `duration=` horizons, and the `index=` stamp.
