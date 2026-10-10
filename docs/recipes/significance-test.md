---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# Is my backtest significant?

`position_test` compares a strategy's result with the results of randomly timed
copies of the same strategy, replayed through the same backtest engine. The
copies keep the trades (their number, sides, sizes, and holding periods) and
change only when they happen. The p-value is the share of copies that do at
least as well as the real timing. This recipe tests a short-term trend rule
on two markets: the real BTC tape, and a synthetic market with built-in
momentum where the rule has an edge.

## The real tape

```{code-cell} ipython3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from screamer import Resample, EwMean, LogReturn, BacktestOHLCTarget, position_test

trades = pd.read_csv("../notebooks/data/deribit_btc_perp_6h.csv")
bars, _ = Resample(freq=60_000, agg="ohlc")(
    trades["price"].to_numpy(), trades["timestamp"].to_numpy())
o, h, l, c = bars.T

def trend(close):
    # long after recent gains, short after recent losses
    return np.sign(np.nan_to_num(EwMean(span=10)(LogReturn()(close))))

signal = trend(c)
engine = BacktestOHLCTarget(taker_fee=5e-4)

real = position_test(engine, signal, o, h, l, c, n=999, metric="sharpe", seed=0)
print(f"observed Sharpe per bar {real.observed:+.3f}, p = {real.p_value:.3f}")
```

## A market with momentum

The synthetic market has autocorrelated returns, so a trend rule has an edge by
construction. The same rule and engine run on it.

```{code-cell} ipython3
rng = np.random.default_rng(1)
n = 2000
r = np.zeros(n)
e = rng.standard_normal(n) * 0.002
for i in range(1, n):
    r[i] = 0.2 * r[i - 1] + e[i]          # returns with autocorrelation 0.2
mc = 30_000 * np.exp(np.cumsum(r))
mo = np.r_[mc[0], mc[:-1]]
mh, ml = np.maximum(mo, mc), np.minimum(mo, mc)

msignal = trend(mc)
momentum = position_test(engine, msignal, mo, mh, ml, mc, n=999, metric="sharpe", seed=0)
print(f"observed Sharpe per bar {momentum.observed:+.3f}, p = {momentum.p_value:.3f}")
```

Each histogram is the null distribution: the Sharpe ratio of the 999 randomly
timed copies. The red line is the real timing. On the BTC tape it sits inside
the null, so the result is consistent with luck; on the momentum market it lies
beyond every copy. Nearly every BTC copy has a negative Sharpe ratio because the
copies pay the same fees as the real strategy, so the test measures the timing
and not whether the strategy made money.

```{code-cell} ipython3
fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
for ax, res, title in [(axes[0], real, "BTC tape"), (axes[1], momentum, "momentum market")]:
    ax.hist(res.null, bins=40, color="lightsteelblue", label="random timing")
    ax.axvline(res.observed, color="crimson", lw=2, label=f"actual, p = {res.p_value:.3f}")
    ax.set_title(title)
    ax.set_xlabel("Sharpe per bar")
    ax.legend(fontsize=8)
plt.tight_layout()
```

A p-value is only valid for a strategy chosen before looking at the data. If you
tried several variants and kept the best, see the note on selection in the
[`position_test`](../functions_fin/position_test.md) page.

Related pages: [`position_test`](../functions_fin/position_test.md),
[`circular_shift`](../functions_fin/circular_shift.md),
[`segment_shuffle`](../functions_fin/segment_shuffle.md), and
[`BacktestOHLCTarget`](../functions_fin/BacktestOHLCTarget.md).
