---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# Labels and training sets

A label, or target, is the value a supervised model learns to predict from the
features. This page shows how to build the common targets from screamer
operators and how [`forecast_pairs`](functions_supervised/forecast_pairs.md)
pairs each one with the features it follows.

## Training rows and live rows

A training row holds the features known at step `t` next to what happened
between `t` and `t + h`. Live, the same row exists with the target still
unknown:

| | features | target |
|---|---|---|
| training, row t | `X[t]` | outcome of `(t, t + h]` |
| live, row t | `X[t]` | NaN, not known yet |

The feature matrix is built by the same code, at the same timestamps, in
research and live, and the model learns "features now, outcome next".

## Stamping a target

A target is a causal series, stamped when its horizon ends: its value at step
`t` uses data up to `t` only. `LogReturn(h)(close)` at step `t` is the return
over `(t - h, t]`. `forecast_pairs(X, y, count=h)` moves that value back to the
row of the features at the start of the horizon, so row `t` holds `X[t]` and
the return over `(t, t + h]`. The last `h` targets are NaN: those are the rows
a live model sees.

The examples on this page use a simulated price with `h = 5` and a feature
matrix of three columns: the last return, the return over the last `h` steps,
and the volatility expected over the next `h` steps, from the standard
deviation of the last 20 one-step returns.

```{code-cell} ipython3
import numpy as np
from screamer import (LogReturn, RollingStd, RollingMax, RollingMin, Lag, Sign,
                      CrossRank, TripleBarrier, TripleBarrierOHLC, forecast_pairs)

rng = np.random.default_rng(7)
n, h = 2000, 5
close = 100 * np.exp(np.cumsum(0.01 * rng.standard_normal(n)))

r1 = LogReturn(1)(close)
X = np.column_stack([
    r1,                                  # return over the last step
    LogReturn(h)(close),                 # return over the last h steps
    RollingStd(20)(r1) * np.sqrt(h),     # volatility expected over the next h steps
])

y = LogReturn(h)(close)
Xp, yp = forecast_pairs(X, y, count=h)
print(np.column_stack([Xp, yp])[-7:].round(4))
```

Pass `dropna=True` to keep only the rows where the features and the target are
known.

## Choosing a target

| I want to predict | Target |
|---|---|
| [how much the price moves](#how-much-the-price-moves) | `LogReturn(h)(close)` |
| [up or down](#up-or-down) | `Sign` of that return |
| [up, down, or flat beyond a dead zone](#up-down-or-flat-beyond-a-dead-zone) | the return against a threshold, applied after pairing |
| [a move in units of expected volatility](#a-move-in-units-of-expected-volatility) | the paired return divided by a volatility column of `X` |
| [how volatile the next period is](#how-volatile-the-next-period-is) | `RollingStd(h)` of one-step returns |
| [how far it runs up and down](#how-far-it-runs-up-and-down) | `RollingMax(h)` and `RollingMin(h)` of the price, over the price `h` steps ago |
| [which of two price levels is reached first](#which-of-two-price-levels-is-reached-first) | `TripleBarrier` or `TripleBarrierOHLC` |
| [which assets do best](#which-assets-do-best) | `LogReturn(h)` on a `(T, K)` panel, then `CrossRank` |

### How much the price moves

[`LogReturn(h)`](functions_fin/LogReturn.md) is the log return over the last
`h` steps. Paired with `count=h`, it is the return over the next `h` steps.

```{code-cell} ipython3
y = LogReturn(h)(close)
Xp, yp = forecast_pairs(X, y, count=h, dropna=True)
print(f"{len(yp)} rows, mean target {yp.mean():+.5f}, std {yp.std():.5f}")
```

### Up or down

[`Sign`](functions_math/Sign.md) of the return gives `+1`, `-1`, or `0` for a
return of exactly zero.

```{code-cell} ipython3
y = Sign()(LogReturn(h)(close))
Xp, yp = forecast_pairs(X, y, count=h, dropna=True)
print(np.unique(yp, return_counts=True))
```

### Up, down, or flat beyond a dead zone

A return whose size is within a dead zone `dz` is labeled `0`. The threshold
is applied to the paired return.

```{code-cell} ipython3
dz = 0.01
Xp, ret = forecast_pairs(X, LogReturn(h)(close), count=h, dropna=True)
yp = np.sign(ret) * (np.abs(ret) > dz)
print(np.unique(yp, return_counts=True))
```

### A move in units of expected volatility

Dividing the paired return by a volatility column of `X` expresses each move in
units of the volatility expected over the horizon. Each row is stamped at the
features' step, so the volatility in its `X` row is the one known when the
horizon starts, and no lag is needed. The dead zone above can then be set in
the same units.

```{code-cell} ipython3
Xp, ret = forecast_pairs(X, LogReturn(h)(close), count=h, dropna=True)
z = ret / Xp[:, 2]                       # column 2: expected volatility over h steps
yp = np.sign(z) * (np.abs(z) > 0.5)      # flat within half an expected move
print(np.unique(yp, return_counts=True))
```

### How volatile the next period is

[`RollingStd(h)`](functions_rolling/RollingStd.md) of the one-step returns is
their standard deviation over the last `h` steps. Paired with `count=h`, it is
the realized volatility of the next `h` steps.

```{code-cell} ipython3
y = RollingStd(h)(LogReturn(1)(close))
Xp, yp = forecast_pairs(X, y, count=h, dropna=True)
print(f"mean realized volatility over {h} steps: {yp.mean():.5f}")
```

### How far it runs up and down

[`RollingMax(h)`](functions_rolling/RollingMax.md) and
[`RollingMin(h)`](functions_rolling/RollingMin.md) of the price, divided by the
price `h` steps ago from [`Lag(h)`](functions_misc/Lag.md), give the largest
rise and the largest fall over `(t - h, t]` relative to the price at `t - h`.
Paired with `count=h`, they measure the next `h` steps from the price at the
features' step. Each target is one column, so each is paired on its own.

```{code-cell} ipython3
start = Lag(h)(close)
up = RollingMax(h)(close) / start - 1
down = RollingMin(h)(close) / start - 1
Xp, up_p = forecast_pairs(X, up, count=h, dropna=True)
_, down_p = forecast_pairs(X, down, count=h, dropna=True)
print(f"median run up {np.median(up_p):+.4f}, median run down {np.median(down_p):+.4f}")
```

### Which of two price levels is reached first

[`TripleBarrier`](functions_fin/TripleBarrier.md) labels each step `+1` when
the price reaches the upper barrier first, `-1` when it reaches the lower
barrier first, and `0` when it reaches neither within `horizon` steps. Its
output at step `t` describes the entry at `t - horizon`, so use
`count=horizon` to pair it with the features of its entry step. The output has
three columns, `label`, `return`, and `holding`; pair each one you need.

```{code-cell} ipython3
out = TripleBarrier(horizon=h, upper=1.0, lower=1.0, barrier_unit="vol", vol_span=20)(close)
Xp, label = forecast_pairs(X, out[:, 0], count=h, dropna=True)
_, holding = forecast_pairs(X, out[:, 2], count=h, dropna=True)
print(np.unique(label, return_counts=True))
print(f"mean holding: {holding.mean():.2f} steps")
```

The label of the entry at step `e` spans `[e, e + holding]`, the interval a
purged cross-validation needs for each row.

[`TripleBarrierOHLC`](functions_fin/TripleBarrierOHLC.md) tests the barriers
on each bar's open, high, and low. When both barriers lie inside one bar, the
order is unknown and the entry is NaN, so `dropna=True` leaves it out.

```{code-cell} ipython3
open_ = np.r_[close[0], close[:-1]]
high = np.maximum(open_, close) * np.exp(np.abs(rng.standard_normal(n)) * 0.003)
low = np.minimum(open_, close) * np.exp(-np.abs(rng.standard_normal(n)) * 0.003)

out = TripleBarrierOHLC(horizon=h, upper=1.0, lower=1.0, barrier_unit="vol",
                        vol_span=20)(open_, high, low, close)
Xp, label = forecast_pairs(X, out[:, 0], count=h, dropna=True)
print(np.unique(label, return_counts=True))
```

### Which assets do best

On a `(T, K)` panel, one column per asset, `LogReturn(h)` gives each asset's
return over the last `h` steps, and [`CrossRank`](functions_cross/CrossRank.md)
ranks them against each other at each step, from `-1` for the worst to `1` for
the best. `y` is one column, so each asset is paired with its own features,
and the rows of all assets are stacked into one training set.

```{code-cell} ipython3
K = 5
prices = 100 * np.exp(np.cumsum(0.01 * rng.standard_normal((n, K)), axis=0))
features = LogReturn(20)(prices)          # each asset's return over the last 20 steps
rank = CrossRank()(LogReturn(h)(prices))  # rank of each asset's return over the last h steps

pairs = [forecast_pairs(features[:, k], rank[:, k], count=h, dropna=True) for k in range(K)]
Xp = np.concatenate([p[0] for p in pairs])
yp = np.concatenate([p[1] for p in pairs])
print(f"{len(yp)} rows from {K} assets")
```
