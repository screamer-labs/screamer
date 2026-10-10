---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# A parameter grid and its robustness

This recipe backtests 25 variants of a trend rule in one engine call, plots
their Sharpe ratios as a heatmap, and then asks two questions about the best
cell: is its timing better than chance once the selection over 25 variants is
accounted for, and how much of its Sharpe ratio would the best of 25 random
variants reach anyway. The data is daily Binance BTC/USDT bars from January
2022 to September 2026.

## The grid as one target matrix

Each column of `target` is one variant: long one BTC when the fast exponential
average of the close is above the slow one, short when it is below. A backtest
engine given a `(T, K)` target and `(T, K)` market inputs runs `K` independent
backtests and returns a `(T, K, 4)` array.

```{code-cell} ipython3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from screamer import (EwMean, Sign, FillNa, BacktestOHLCTarget,
                      position_test, deflated_sharpe)

bars = pd.read_csv("../notebooks/data/binance_daily_20.csv", parse_dates=["date"])
btc = bars[bars["symbol"] == "BTC"]
o, h, l, c = (btc[k].to_numpy() for k in ["open", "high", "low", "close"])

fast = [5, 10, 20, 30, 50]
slow = [60, 100, 150, 200, 300]
grid = [(f, s) for f in fast for s in slow]
target = np.column_stack([Sign()(FillNa(0.0)(EwMean(span=f)(c) - EwMean(span=s)(c)))
                          for f, s in grid])

T, K = target.shape
engine = BacktestOHLCTarget(taker_fee=1e-3, slippage=5e-4)
out = engine(target, *(np.broadcast_to(x[:, None], (T, K)) for x in (o, h, l, c)))
out.shape
```

## Sharpe ratios and the family-wise test

`position_test` accepts the same `(T, K)` target. Each of its 999 randomized
rounds shifts all 25 columns by one common random offset, and the result holds
an ordinary p-value per variant and a `family_p_value` that compares each
variant with the best of the 25 shifted variants in that round. Its `observed`
field is each variant's Sharpe ratio per day.

```{code-cell} ipython3
res = position_test(engine, target, o, h, l, c, n=999, seed=0)
sharpe = res.observed * np.sqrt(365)                 # annualized

best = int(np.argmax(res.observed))
print(f"best variant: fast {grid[best][0]}, slow {grid[best][1]}, "
      f"Sharpe {sharpe[best]:.2f}")
print(f"p-value {res.p_value[best]:.3f}, family p-value {res.family_p_value[best]:.3f}")
```

The heatmap shows a positive Sharpe ratio in every cell, from 0.14 to 0.72, and
mostly higher in the 300-day column. The best cell, a 50-day average against a 200-day one, has a
p-value of 0.007 on its own. Compared with the best of 25 randomly timed
variants, its family p-value is 0.051.

```{code-cell} ipython3
fig, ax = plt.subplots(figsize=(6, 4.5))
im = ax.imshow(sharpe.reshape(len(fast), len(slow)), cmap="viridis", origin="lower")
ax.set_xticks(range(len(slow)), slow)
ax.set_yticks(range(len(fast)), fast)
ax.set_xlabel("slow span (days)")
ax.set_ylabel("fast span (days)")
for i, j in np.ndindex(len(fast), len(slow)):
    ax.text(j, i, f"{sharpe[i * len(slow) + j]:.2f}", ha="center", va="center",
            color="white" if sharpe[i * len(slow) + j] < 0.5 else "black", fontsize=8)
fig.colorbar(im, label="annualized Sharpe")
plt.tight_layout()
```

## Deflated Sharpe ratio

`deflated_sharpe` takes the best variant's pnl and the per-day Sharpe ratios of
every variant tried. From their number and spread it estimates the Sharpe ratio
the best of 25 variants would reach by luck, and returns the probability that
the best variant's true Sharpe ratio is above that benchmark.

```{code-cell} ipython3
prob, benchmark = deflated_sharpe(out[:, best], res.observed)
print(f"luck benchmark {benchmark * np.sqrt(365):.2f} annualized, "
      f"probability above it {prob:.2f}")
```

The best of 25 variants with no skill would reach an annualized Sharpe ratio of
about 0.30, and the probability that the 50/200 rule's true Sharpe ratio exceeds
it is 0.82. Both corrections lead to the same reading: the grid as a whole leans
positive, but the best cell's Sharpe ratio is not clearly above what selecting
the best of 25 produces by chance.

Related pages: [`position_test`](../functions_fin/position_test.md),
[`deflated_sharpe`](../functions_fin/deflated_sharpe.md),
[`probabilistic_sharpe`](../functions_fin/probabilistic_sharpe.md), and the
[significance recipe](significance-test.md).
