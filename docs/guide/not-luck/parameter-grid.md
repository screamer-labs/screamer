---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# Check that a result holds across nearby parameters

A `(T, K)` target runs `K` variants of a rule through one `BacktestOHLCTarget`
call; `position_test` gives the best variant a p-value corrected for the choice
among `K`, and `deflated_sharpe` the probability that its Sharpe ratio beats
the best that `K` variants would reach by luck.

```{code-cell} ipython3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from screamer import (EwMean, Sign, BacktestOHLCTarget,
                      position_test, deflated_sharpe)

df = pd.read_csv("../../notebooks/data/binance_daily_20.csv")
btc = df[df["symbol"] == "BTC"]
o, h, l, c = (btc[k].to_numpy() for k in ["open", "high", "low", "close"])

fast, slow = [5, 10, 20, 30, 50], [60, 100, 150, 200, 300]
grid = [(f, s) for f in fast for s in slow]
ema = {n: EwMean(span=n)(c) for n in fast + slow}
target = np.column_stack([Sign()(ema[f] - ema[s]) for f, s in grid])
engine = BacktestOHLCTarget(taker_fee=1e-3, slippage=5e-4)

out = engine(target, o, h, l, c)
res = position_test(engine, target, o, h, l, c, n=999)
best = int(np.argmax(res.observed))
prob, benchmark = deflated_sharpe(out[:, best], res.observed)
print(f"best {grid[best]}: p-value {res.p_value[best]:.3f}, "
      f"family p-value {res.family_p_value[best]:.3f}\n"
      f"deflated Sharpe {prob:.2f}, benchmark {benchmark * np.sqrt(365):.2f}")
```

```{code-cell} ipython3
sharpe = (res.observed * np.sqrt(365)).reshape(len(fast), len(slow))
fig, ax = plt.subplots(figsize=(6, 4.5))
im = ax.imshow(sharpe, cmap="viridis", origin="lower")
ax.set_xticks(range(len(slow)), slow)
ax.set_yticks(range(len(fast)), fast)
ax.set_xlabel("slow span (days)")
ax.set_ylabel("fast span (days)")
for i, j in np.ndindex(sharpe.shape):
    ax.text(j, i, f"{sharpe[i, j]:.2f}", ha="center", va="center", fontsize=8,
            color="white" if sharpe[i, j] < 0.5 else "black")
fig.colorbar(im, label="annualized Sharpe")
plt.tight_layout()
```

Each of the 25 columns of `target` is one variant of a trend rule on daily
BTC: long one BTC while a fast exponential average of the close is above a
slow one, short while it is below, for every pair `(fast, slow)` in `grid`.
The engine returns a `(T, 25, 4)` array, one backtest per column, with a 0.1%
taker fee and 0.05% slippage. `res.observed` is each variant's Sharpe ratio
per day.

Every cell has a positive Sharpe ratio, from 0.14 to 0.72, so the result does
not rest on one setting. The best cell, 50 against 200 days, has a p-value of
0.007 on its own and a family p-value of 0.051 after the choice among 25. Its
deflated Sharpe ratio of 0.82 is the probability that its true Sharpe ratio is
above the 0.30 per year that the best of 25 variants with no skill would reach.
The grid leans positive, but the best cell is not clearly better than what
picking the best of 25 produces by chance.

## Go deeper

- [Choose parameters with walk-forward testing](walk-forward.md): pick a variant on past data only and trade it out of sample.
- [Test whether a strategy's timing beats chance](timing-test.md): the test for one variant.
- [`position_test`](../../functions_fin/position_test.md), [`deflated_sharpe`](../../functions_fin/deflated_sharpe.md): the family-wise test and the luck benchmark. Every round of `position_test` shifts all 25 columns by the same random offset, and `family_p_value` compares the best variant with the best of the 25 shifted copies.
