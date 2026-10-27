---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# Choose parameters with walk-forward testing

`walk_forward` picks the best of several strategy variants on each training
window, trades that pick on the test window that follows, and reports the
stitched out-of-sample backtest.

```{code-cell} ipython3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from screamer import (EwMean, LogReturn, Sign, FillNa, BacktestOHLCTarget,
                      backtest_report, walk_forward)

df = pd.read_csv("../../notebooks/data/binance_daily_20.csv",
                 parse_dates=["date"])
btc = df[df["symbol"] == "BTC"].set_index("date")
o, h, l, c = (btc[k].to_numpy() for k in ["open", "high", "low", "close"])

spans, ret = [5, 10, 20, 40, 80, 160], LogReturn()(c)
X = np.column_stack([Sign()(FillNa(0.0)(EwMean(span=s)(ret))) for s in spans])
engine = BacktestOHLCTarget(taker_fee=1e-3, slippage=5e-4)

wf = walk_forward(engine, X, o, h, l, c, train=365, test=91, periods_per_year=365)
print(f"walk-forward Sharpe {wf.stats['sharpe']:.2f}, column per fold", wf.choice)

# Scoring only, looks ahead: the span that is best over the whole history
full, t0 = engine(X, o, h, l, c), wf.fold_start[0]
sharpes = [backtest_report(full[:, k])[1]["sharpe"] for k in range(len(spans))]
best = int(np.argmax(sharpes))
_, hindsight = backtest_report(full[t0:, best], periods_per_year=365)
print(f"hindsight span {spans[best]}: Sharpe {hindsight['sharpe']:.2f}")
```

```{code-cell} ipython3
dates = btc.index[t0:]
fig, ax = plt.subplots(figsize=(9, 3.5))
ax.plot(dates, full[t0:, best, 0] - full[t0 - 1, best, 0], color="0.6",
        label=f"hindsight best (span {spans[best]})")
ax.plot(dates, wf.output[t0:, 0] - wf.output[t0 - 1, 0], color="crimson",
        label="walk-forward")
ax.axhline(0, color="k", lw=0.5)
ax.set_ylabel("profit (USD, 1 BTC)")
ax.legend(loc="upper left", fontsize=8)
plt.tight_layout()
```

Each column of `X` is one variant of a trend rule on daily BTC: long while an
exponential mean of the daily log return is positive, short while it is
negative, with spans from 5 to 160 days. Every fold chooses the span with the
best Sharpe ratio over the previous 365 days (`train`) and trades it for the
next 91 days (`test`), so the first test day is 2023-01-01. Every fill pays a
0.1% taker fee and 0.05% slippage.

`wf.choice` holds the column of `X` chosen in each of the 16 folds: every one
of the six spans is chosen at least once, and the span changes 10 times. From
the first test day on, the walk-forward run has an annualized Sharpe ratio of
0.04. Span 40, the best over the whole history, has 0.50 over the same days,
but choosing it used the years it is scored on. The walk-forward number is the
estimate of what "trade the span that did best last year" delivers.

## Go deeper

- [Check that a result holds across nearby parameters](parameter-grid.md): how the Sharpe ratio varies across a grid of variants.
- [I tuned my parameters. Is the result real?](../../notebooks/21-is-the-tuned-result-real.ipynb): test a whole parameter search against permuted markets.
- [`walk_forward`](../../functions_fin/walk_forward.md): anchored windows, other metrics, portfolio engines, and funding with `carry=`.
- [`BacktestCarry`](../../functions_fin/BacktestCarry.md): the funding or carry that `carry=` applies to every candidate.
