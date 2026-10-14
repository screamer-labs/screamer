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
from screamer import (Resample, EwMean, LogReturn, Sign, FillNa, BacktestOHLCTarget,
                      position_test, probabilistic_sharpe, deflated_sharpe)

trades = pd.read_csv("../notebooks/data/deribit_btc_perp_6h.csv")
bars, _ = Resample(freq=60_000, agg="ohlc")(
    trades["price"].to_numpy(), trades["timestamp"].to_numpy())
o, h, l, c = bars.T

def trend(close, span=10):
    # long after recent gains, short after recent losses, flat during warm-up
    return Sign()(FillNa(0.0)(EwMean(span=span)(LogReturn()(close))))

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

## Is the Sharpe ratio above zero?

`position_test` asks whether the timing beats random timing. Whether the
strategy makes money after costs is a separate question.
`probabilistic_sharpe` answers it analytically: it returns the probability that
the true Sharpe ratio is above a benchmark (0 by default), given the observed
Sharpe ratio, the number of steps, and the skewness and kurtosis of the pnl.

```{code-cell} ipython3
out = engine(msignal, mo, mh, ml, mc)
print(f"P(true Sharpe > 0) = {probabilistic_sharpe(out):.3f}")
```

On the momentum market the timing beats random timing (p = 0.001), but after
5 bps per trade the Sharpe ratio per bar is small, and the probability that it
is above zero is 0.89.

## Several variants

A p-value is valid only for a strategy chosen before looking at the data. If
several variants were tried and the best one is reported, the choice is itself
a source of luck. Passing a `(T, K)` target, one column per variant, to
`position_test` shifts all variants by the same offset in every round and
returns `family_p_value`, which compares each variant with the best of the `K`
shifted results (the max statistic of White's Reality Check).
`deflated_sharpe` is the analytic counterpart: it raises the benchmark of
`probabilistic_sharpe` to the Sharpe ratio the best of the tried variants would
reach by luck.

```{code-cell} ipython3
spans = [4, 6, 8, 12, 16, 24, 32, 48]
targets = np.column_stack([trend(mc, span) for span in spans])
family = position_test(engine, targets, mo, mh, ml, mc, n=999, metric="sharpe", seed=0)
best = int(np.argmax(family.observed))
print(f"best span {spans[best]}: p = {family.p_value[best]:.3f}, "
      f"family p = {family.family_p_value[best]:.3f}")

runs = engine(targets, mo, mh, ml, mc)            # (T, K, 4), one backtest per span
pnl = runs[:, :, 1]
sharpes = pnl.mean(axis=0) / pnl.std(axis=0, ddof=1)
p, benchmark = deflated_sharpe(runs[:, best], sharpes)
print(f"deflated Sharpe: P(true Sharpe > {benchmark:.4f}) = {p:.3f}")
```

The best span keeps a family p-value of 0.022, so its timing still beats random
timing after the choice among eight. The deflated Sharpe ratio asks the
stricter question of whether the net Sharpe ratio beats luck: the best
variant's 0.026 per bar is only a little above the 0.020 that the best of these
eight would reach by chance, which gives a probability of 0.6.

The notebook [Is it skill? Choosing a randomization test](../notebooks/20-randomization-tests.ipynb)
compares `position_test` with `strategy_test`, which reruns the whole strategy
on permuted markets, and shows why a strategy tuned on the data needs the
latter.

Related pages: [`position_test`](../functions_fin/position_test.md),
[`strategy_test`](../functions_fin/strategy_test.md),
[`probabilistic_sharpe`](../functions_fin/probabilistic_sharpe.md),
[`deflated_sharpe`](../functions_fin/deflated_sharpe.md),
[`circular_shift`](../functions_fin/circular_shift.md),
[`segment_shuffle`](../functions_fin/segment_shuffle.md), and
[`BacktestOHLCTarget`](../functions_fin/BacktestOHLCTarget.md).
