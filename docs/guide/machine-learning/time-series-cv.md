---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# Cross-validate on time series without leakage (purged time series split)

`time_splits` cuts time-ordered rows into folds that train only on rows before
their test block and leave out each training row whose label reaches into it;
scikit-learn takes the folds as `cv=`.

```{code-cell} ipython3
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from screamer import (LogReturn, RollingZscore, RollingStd, TripleBarrierOHLC,
                      forecast_pairs, time_splits)

df = pd.read_csv("../../notebooks/data/binance_daily_20.csv")
btc = df[df["symbol"] == "BTC"]
o, h, l, c = (btc[k].to_numpy() for k in ["open", "high", "low", "close"])

X = np.column_stack([LogReturn(20)(c), RollingZscore(50)(c),
                     RollingStd(20)(LogReturn(1)(c))])
tb = TripleBarrierOHLC(horizon=10, upper=1.0, lower=1.0,
                       barrier_unit="vol", vol_span=20)(o, h, l, c)
Xs, ys = forecast_pairs(X, tb[:, 0], count=10, dropna=True)
cv = time_splits(len(Xs), label_length=10)
for model in [LogisticRegression(), DummyClassifier()]:
    print(type(model).__name__, cross_val_score(model, Xs, ys, cv=cv).round(3))
```

```{code-cell} ipython3
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(9, 2.8))
for f, (train, test) in enumerate(cv):
    purged = np.arange(train[-1] + 1, test[0])
    for rows, color in [(train, "steelblue"), (purged, "orange"),
                        (test, "crimson")]:
        ax.scatter(rows, np.full(len(rows), f), color=color, marker="|", s=60)
ax.set_yticks(range(len(cv)), [f"fold {f + 1}" for f in range(len(cv))])
ax.invert_yaxis()
ax.set_xlabel("training row (blue train, orange left out, red test)")
plt.tight_layout()
```

The rows are those of [Pair features with labels without look-ahead](training-set.md):
three features of daily BTC known at each close, and the triple-barrier label of
the next 10 days, `+1`, `-1`, or `0`. `time_splits` makes five folds. Each tests
on a block of about 280 rows and trains on the rows before it. A label covers up
to 10 days after its row, so the 10 training rows just before each test block
(orange) have labels that use prices from inside it, and `label_length=10`
leaves them out. Without that gap, a model could score well on the first days of
a test block by having seen their prices in its training labels.

The logistic regression scores the same as `DummyClassifier`, which always
predicts the most common label of its training rows, in four folds of five, and
worse in the first. These three features do not predict the label, and the
purged folds show it. `label_length` also takes one value per row, such as the
holding period that `TripleBarrierOHLC` returns in its third column, which
leaves out fewer rows.

## Go deeper

- [Pair features with labels without look-ahead](training-set.md): build `Xs` and `ys`, and purge with each label's own holding period.
- [Choose parameters with walk-forward testing](../not-luck/walk-forward.md): the same idea for choosing a strategy variant.
- [Classifying direction](../../notebooks/23-classifying-direction.ipynb): train and score a classifier on direction labels.
- [`time_splits`](../../functions_fin/time_splits.md): test block size, a cap on the training rows, and the scikit-learn equivalence.
