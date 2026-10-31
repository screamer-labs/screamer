---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# Run a Monte Carlo permutation test on a whole strategy

`strategy_test` runs a whole strategy, parameter search included, on the real
bars and on many permuted copies of them, and returns the share of permuted
markets on which it does at least as well, a p-value.

```{code-cell} ipython3
import numpy as np
import pandas as pd
from screamer import (EwMean, LogReturn, Sign, FillNa, BacktestOHLCTarget,
                      backtest_report, strategy_test)

df = pd.read_csv("../../notebooks/data/binance_daily_20.csv")
btc = df[df["symbol"] == "BTC"]
o, h, l, c = (btc[k].to_numpy() for k in ["open", "high", "low", "close"])
engine = BacktestOHLCTarget(taker_fee=1e-3, slippage=5e-4)
spans = [5, 10, 20, 40, 80, 160]

def strategy(o, h, l, c):
    ret = LogReturn()(c)
    X = np.column_stack([Sign()(FillNa(0.0)(EwMean(span=s)(ret))) for s in spans])
    out = engine(X, o, h, l, c)
    sharpe = [backtest_report(out[:, k])[1]["sharpe"] for k in range(len(spans))]
    return out[:, int(np.argmax(sharpe))]

res = strategy_test(strategy, o, h, l, c, n=999, periods_per_year=365)
print(f"Sharpe {res.observed:.2f}, p-value {res.p_value:.3f}")
```

```{code-cell} ipython3
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(8, 3.5))
ax.hist(res.null, bins=40, color="lightsteelblue",
        label="best span on a permuted market")
ax.axvline(res.observed, color="crimson", lw=2,
           label=f"best span on the real market, p = {res.p_value:.3f}")
ax.set_xlabel("annualized Sharpe ratio")
ax.legend(fontsize=8)
plt.tight_layout()
```

`strategy` tries six variants of a trend rule on daily BTC (long while an
exponential mean of the daily log return is positive, short while it is
negative, with spans from 5 to 160 days) and returns the backtest of the one
with the best Sharpe ratio. Every fill is at the next open and pays a 0.1%
taker fee and 0.05% slippage. With the default `method="bars"`, every permuted
market keeps the first bar, the last close, and the distribution of daily gaps
and bar shapes, in a random order (see
[Generate synthetic price paths by shuffling bars or returns (permuted bars, surrogate data)](synthetic-paths.md)).
The strategy repeats its search on each one, so the null distribution is that
of the best of six spans on a market with no serial structure.

Picking the best span on the real bars gives a Sharpe ratio of 0.44, and the
same search does at least as well on 22% of the permuted markets. The trend it
finds in BTC is no stronger than what it finds in markets that have none. This is
the Monte Carlo permutation test of Masters (2020). Unlike
[`position_test`](timing-test.md), which keeps one fixed position series, it
also accounts for everything the strategy fits or selects.

## Go deeper

- [Choose parameters with walk-forward testing](walk-forward.md): the same six spans, chosen out of sample.
- [Test a signal against time-shifted copies of itself (circular shift test)](shifted-copies.md): keep the market and rotate a feature instead.
- [I tuned my parameters. Is the result real?](../../notebooks/21-is-the-tuned-result-real.ipynb): a larger search, and its in-sample and permuted results side by side.
- [Does my feature carry information?](../../notebooks/22-does-the-feature-carry-information.ipynb): `method="shift"` for a model's features.
- [`strategy_test`](../../functions_fin/strategy_test.md): `method="returns"` and `"shift"`, volume, panels, and portfolio engines.
