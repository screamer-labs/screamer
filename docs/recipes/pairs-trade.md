---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# A pairs trade

A pairs trade holds one asset against another in the ratio that cancels their
common movement, and bets that the remaining spread returns to its average.
This recipe trades ETH against BTC on six hours of one-minute bars: a Kalman
filter estimates the hedge ratio, a rolling ADF test checks that the spread
still reverts, `BandSignal` sets the position, and
`BacktestPortfolioOHLCTarget` runs both legs from one account with fees. A
randomization test then asks whether the result is more than luck. The
estimators themselves are compared in the notebook
[Hedge ratios and spread estimation](../notebooks/18-pairs-and-mean-reversion.ipynb).

## Bars

`Resample(agg="ohlc")` turns each trade tape into one-minute bars with columns
open, high, low, close. `fill="nan"` writes a NaN row for a minute without
trades between two trades, and `CombineLatest` aligns the eight columns of the
two assets on their timestamps. Each NaN row becomes a flat bar at the last
traded price, so an order on that minute fills at that price.

```{code-cell} ipython3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from screamer import (Resample, CombineLatest, Ffill, KalmanRegression,
                      RollingZscore, RollingADF, BandSignal,
                      BacktestPortfolioOHLCTarget, position_test)

def bars(path):
    trades = pd.read_csv(path)
    return Resample(freq=60_000, agg="ohlc", fill="nan")(
        trades["price"].to_numpy(), trades["timestamp"].to_numpy())

btc, btc_t = bars("../notebooks/data/deribit_btc_perp_6h.csv")
eth, eth_t = bars("../notebooks/data/deribit_eth_perp_6h.csv")
cols, ts = CombineLatest()(*btc.T, *eth.T, index=[btc_t] * 4 + [eth_t] * 4)

# (T, 2) arrays, column 0 = BTC, column 1 = ETH
close = Ffill()(cols[:, [3, 7]])                     # last traded price
open_, high, low = (np.where(np.isnan(x), close, x)  # flat bar on a minute with no trade
                    for x in (cols[:, [0, 4]], cols[:, [1, 5]], cols[:, [2, 6]]))
t = pd.to_datetime(ts, unit="ms")
log_btc, log_eth = np.log(close[:, 0]), np.log(close[:, 1])
close.shape
```

## Hedge ratio and spread

`KalmanRegression(delta, obs_var)(log_eth, log_btc)` regresses ETH on BTC with
an intercept and slope that may drift slowly. Column 0 is the hedge ratio and
column 2 the spread, the prediction error of the previous step's estimate.
`delta=1e-10` keeps the intercept slow: with a much larger `delta` it follows
the spread itself, and the spread then passes the ADF test even for unrelated
assets. `RollingZscore` expresses the spread in standard deviations of the last
30 minutes.

```{code-cell} ipython3
kf = KalmanRegression(delta=1e-10, obs_var=1e-6)(log_eth, log_btc)
beta, spread = kf[:, 0], kf[:, 2]
z = RollingZscore(30)(spread)
adf = RollingADF(window_size=60, lags=1)(spread)
```

## Position and legs

`BandSignal(entry=2.0, exit=0.0, stop=3.5)` goes short the spread when its
z-score reaches 2, long at -2, and flat when it crosses back through 0. The
position is held only while the rolling ADF statistic is below -3.34, the 5%
Engle-Granger critical value (the hedge ratio comes from the same prices), so
the trade is off when the spread stops reverting.

One unit of spread is long ETH and short `beta` times as much BTC by value; as
weights of equity with a gross exposure of 1, that is `1 / (1 + beta)` in ETH
and `-beta / (1 + beta)` in BTC. The Kalman hedge ratio changes every minute,
so weights sent on every step would rebalance both legs every minute and pay
fees each time. The targets are therefore set only on the minutes where the
position changes, and NaN elsewhere holds the legs.
`BacktestPortfolioOHLCTarget` executes a target decided at a minute's close at
the next minute's open.

```{code-cell} ipython3
position = BandSignal(entry=2.0, exit=0.0, stop=3.5)(z)
position[~(adf < -3.34)] = 0.0
position[:60] = 0.0                                     # warm-up

# columns in the order of the price arrays: BTC, ETH
targets = np.column_stack([-position * beta / (1 + beta), position / (1 + beta)])
changed = np.r_[True, position[1:] != position[:-1]]
targets[~changed] = np.nan                              # NaN holds the position

gross = BacktestPortfolioOHLCTarget()(targets, open_, high, low, close)
net = BacktestPortfolioOHLCTarget(fee=2e-4)(targets, open_, high, low, close)
for name, out in [("no fee", gross), ("2 bps fee", net)]:
    print(f"{name:9s}: {out[:, :, 0].sum(axis=1)[-1] * 1e4:+.1f} bps of capital, "
          f"{int(changed[61:].sum())} position changes")
```

```{code-cell} ipython3
fig, (ax0, ax1, ax2) = plt.subplots(3, 1, figsize=(9, 7), sharex=True)
ax0.plot(t, beta, color="steelblue")
ax0.set_ylabel("hedge ratio")
ax1.plot(t, z, color="0.5", lw=0.8)
ax1.step(t, 2 * position, color="crimson", lw=1.0, where="post")
for level in (2, -2):
    ax1.axhline(level, color="steelblue", lw=0.6, ls="--")
ax1.set_ylabel("z-score, position")
ax2.plot(t, gross[:, :, 0].sum(axis=1) * 1e4, color="0.6", ls="--", label="no fee")
ax2.plot(t, net[:, :, 0].sum(axis=1) * 1e4, color="seagreen", label="2 bps fee")
ax2.axhline(0, color="k", lw=0.5)
ax2.set_ylabel("pnl (bps of capital)")
ax2.legend(fontsize=8, loc="lower left")
plt.tight_layout()
```

Before fees the book ends close to flat. Each position change trades both legs,
so the fee is paid on about the gross exposure every time: twelve changes at
2 bps cost about 24 bps of capital, which is the whole loss.

## Is it more than luck?

`position_test` with a portfolio engine shifts both legs' targets together by a
random offset and reruns the book, keeping the trades and only moving them in
time. The p-value is the share of shifted books that do at least as well as
the real one before fees.

```{code-cell} ipython3
res = position_test(BacktestPortfolioOHLCTarget(), targets, open_, high, low, close,
                    n=999, metric="total_pnl")
print(f"p = {res.p_value:.3f}, real {res.observed * 1e4:+.1f} bps, "
      f"median shifted {np.median(res.null) * 1e4:+.1f} bps")
```

The real book sits in the middle of the shifted ones, so six hours of data give
no evidence that the entry timing carries information.

Related pages: [`KalmanRegression`](../functions_fin/KalmanRegression.md),
[`RollingADF`](../functions_fin/RollingADF.md),
[`BandSignal`](../functions_signal/BandSignal.md),
[`BacktestPortfolioOHLCTarget`](../functions_fin/BacktestPortfolioOHLCTarget.md), and
the [portfolios and pairs guide](../portfolio_backtesting.md).
