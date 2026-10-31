---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# Test a signal against time-shifted copies of itself

`circular_shift` returns copies of a signal rotated by random offsets, which
keep everything about the signal except its alignment with the market; a signal
that predicts returns must do better than most of its copies.

```{code-cell} ipython3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from screamer import LogReturn, circular_shift, forecast_pairs

df = pd.read_csv("../../notebooks/data/binance_daily_20.csv",
                 parse_dates=["date"])
btc = df[df["symbol"] == "BTC"].set_index("date")
c = btc["close"].to_numpy()

signal = LogReturn(20)(c)
X = np.column_stack([signal, circular_shift(signal, 999, min_shift=100, seed=0)])
horizon = 5
Xs, future = forecast_pairs(X, LogReturn(horizon)(c), count=horizon, dropna=True)

ic = np.array([np.corrcoef(column, future)[0, 1] for column in Xs.T])
p_value = (1 + np.sum(ic[1:] >= ic[0])) / len(ic)
print(f"correlation {ic[0]:.3f}, p-value {p_value:.3f}")
```

```{code-cell} ipython3
fig, ax = plt.subplots(figsize=(8, 3.5))
ax.hist(ic[1:], bins=40, color="lightsteelblue", label="shifted copies")
ax.axvline(ic[0], color="crimson", lw=2,
           label=f"signal, p = {p_value:.3f}")
ax.set_xlabel("correlation with the next 5-day return")
ax.legend(fontsize=8)
plt.tight_layout()
```

The signal is the 20-day log return of daily BTC, a momentum signal, and the
question is whether it predicts the log return over the next 5 days.
`forecast_pairs` puts each day's signal next to the return of the 5 days that
follow it and drops the rows where either is unknown. Each of the 999 copies
rotates the signal by a random offset of at least 100 days, with values that
fall off the end wrapping around to the start, so it has the same values and
the same autocorrelation as the signal but sits at the wrong time. A copy
shifted by only a few days would have a 20-day window that overlaps the 5-day
return it is scored on, and a high correlation for that reason alone; `min_shift=100` keeps every copy far from the real alignment. The
leading warmup of 20 NaN values stays in place.

The signal's correlation of -0.018 is near the middle of its copies: 59% of
them do at least as well, so this sample shows no momentum in BTC at this
horizon. The spread of the copies, about -0.08 to 0.07 for the central 90%, is
the range that an uninformative signal with this autocorrelation reaches by
chance. The 5-day returns of neighboring rows overlap, so a textbook standard
error for the correlation would be too small; the copies keep that overlap and
need no correction.

## Go deeper

- [Test whether a strategy's timing beats chance](timing-test.md): rotate a strategy's positions and score them with the backtest.
- [Does my feature carry information?](../../notebooks/22-does-the-feature-carry-information.ipynb): shift the features of a fitted model with `strategy_test(method="shift")`.
- [Pair features with labels without look-ahead](../machine-learning/training-set.md): `forecast_pairs` and horizons.
- [`circular_shift`](../../functions_fin/circular_shift.md): the range of offsets and the NaN warmup.
