---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# Correct a Sharpe ratio for the number of strategies tried (deflated Sharpe ratio)

`deflated_sharpe` gives the probability that the best of the variants you tried
has a true Sharpe ratio above the one the best of that many variants would reach
by luck, and `probabilistic_sharpe` the probability that it is above zero, which
ignores the search.

```{code-cell} ipython3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from screamer import (EwMean, LogReturn, Sign, FillNa, BacktestOHLCTarget,
                      backtest_report, probabilistic_sharpe, deflated_sharpe)

df = pd.read_csv("../../notebooks/data/binance_daily_20.csv")
btc = df[df["symbol"] == "BTC"]
o, h, l, c = (btc[k].to_numpy() for k in ["open", "high", "low", "close"])

spans, ret = np.arange(5, 255, 5), LogReturn()(c)
X = np.column_stack([Sign()(FillNa(0.0)(EwMean(span=s)(ret))) for s in spans])
out = BacktestOHLCTarget(taker_fee=1e-3, slippage=5e-4)(X, o, h, l, c)
sharpes = np.array([backtest_report(out[:, k])[1]["sharpe"]
                    for k in range(len(spans))])
best = int(np.argmax(sharpes))
prob, benchmark = deflated_sharpe(out[:, best], sharpes)
print(f"best span {spans[best]}, Sharpe {sharpes[best] * np.sqrt(365):.2f}\n"
      f"P(true Sharpe > 0) {probabilistic_sharpe(out[:, best]):.2f}\n"
      f"deflated: P(true Sharpe > {benchmark * np.sqrt(365):.2f}) {prob:.2f}")
```

```{code-cell} ipython3
fig, ax = plt.subplots(figsize=(8, 3.5))
ax.bar(spans, sharpes * np.sqrt(365), width=4, color="lightsteelblue")
ax.axhline(benchmark * np.sqrt(365), color="crimson", lw=1.5,
           label="best of 50 by luck")
ax.axhline(0, color="k", lw=0.5)
ax.set_xlabel("span (days)")
ax.set_ylabel("annualized Sharpe ratio")
ax.legend(fontsize=8)
plt.tight_layout()
```

Each of the 50 variants is a trend rule on daily BTC: long one BTC while an
exponential mean of the daily log return is positive, short while it is
negative, with spans from 5 to 250 days. Every fill is at the next open and pays
a 0.1% taker fee and 0.05% slippage. `sharpes` holds each variant's Sharpe ratio
per day, the unit both functions take; the printout annualizes it with the
square root of 365.

The best variant, span 250, has an annualized Sharpe ratio of 0.64, and taken
alone it has a probability of 0.92 of a true Sharpe ratio above zero. That
probability ignores that it was picked from 50. From the number of variants and
the spread of their Sharpe ratios, `deflated_sharpe` puts the best that 50
variants without skill would reach at 0.44 (the red line), and gives the best
variant a probability of 0.67 of beating it. The formula treats the 50 variants
as independent trials. Neighboring spans trade almost the same way, so the
search is worth fewer independent trials than 50, and the benchmark is on the
strict side.

## Go deeper

- [Check that a result holds across nearby parameters](parameter-grid.md): the same correction by resampling, with `position_test`.
- [Choose parameters with walk-forward testing](walk-forward.md): avoid the selection by choosing on past data only.
- [`deflated_sharpe`](../../functions_fin/deflated_sharpe.md), [`probabilistic_sharpe`](../../functions_fin/probabilistic_sharpe.md): the formulas, and the effect of skewness and fat tails.
