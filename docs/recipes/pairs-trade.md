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
still reverts, `BandSignal` sets the position, and `BacktestPortfolioTarget`
runs both legs from one account with fees. A randomization test then asks
whether the result is more than luck.

## Prices

`Resample` turns each trade tape into one-minute closes, and `CombineLatest`
aligns the two assets on their timestamps.

```{code-cell} ipython3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from screamer import (Resample, CombineLatest, KalmanRegression, RollingTLSBeta,
                      RollingZscore, RollingADF, BandSignal, BacktestPortfolioTarget,
                      position_test)

def closes(path):
    trades = pd.read_csv(path)
    return Resample(freq=60_000, agg="last")(
        trades["price"].to_numpy(), trades["timestamp"].to_numpy())

btc, btc_t = closes("../notebooks/data/deribit_btc_perp_6h.csv")
eth, eth_t = closes("../notebooks/data/deribit_eth_perp_6h.csv")
prices, ts = CombineLatest()(btc, eth, index=[btc_t, eth_t])   # columns: BTC, ETH
t = pd.to_datetime(ts, unit="ms")
log_btc, log_eth = np.log(prices[:, 0]), np.log(prices[:, 1])
prices.shape
```

## Hedge ratio and spread

`KalmanRegression(delta, obs_var)(log_eth, log_btc)` regresses ETH on BTC with
an intercept and slope that may drift slowly; a small `delta` keeps the hedge
ratio steady. The spread is what the regression leaves over, and
`RollingZscore` expresses it in standard deviations of the last 30 minutes.
`RollingTLSBeta` gives a second estimate of the hedge ratio that does not
depend on which asset is the dependent one.

```{code-cell} ipython3
kf = KalmanRegression(delta=1e-7, obs_var=1e-6)(log_eth, log_btc)
beta, alpha = kf[:, 0], kf[:, 1]
spread = log_eth - alpha - beta * log_btc
z = RollingZscore(30)(spread)
adf = RollingADF(window_size=60, lags=1)(spread)
tls = RollingTLSBeta(120)(log_eth, log_btc)
```

## Position and legs

`BandSignal(entry=2.0, exit=0.0, stop=3.5)` goes short the spread when its
z-score reaches 2, long at -2, and flat when it crosses back through 0. The
position is held only while the rolling ADF statistic is below -2.86, its 5%
critical value, so the trade is off when the spread stops reverting.

One unit of spread is long ETH and short `beta` times as much BTC by value; as
weights of equity with a gross exposure of 1, that is `1 / (1 + beta)` in ETH
and `-beta / (1 + beta)` in BTC. The targets are set only when the position
changes, and shifted by one minute: `BacktestPortfolioTarget` fills at each
step's price, so a decision made from a minute's close trades at the next
minute's close.

```{code-cell} ipython3
position = BandSignal(entry=2.0, exit=0.0, stop=3.5)(z)
position[~(adf < -2.86)] = 0.0
position[:60] = 0.0                                     # warm-up

weights = np.column_stack([-position * beta / (1 + beta), position / (1 + beta)])
changed = np.r_[True, position[1:] != position[:-1]]
weights[~changed] = np.nan                              # NaN holds the position
targets = np.vstack([np.full((1, 2), np.nan), weights[:-1]])

gross = BacktestPortfolioTarget()(targets, prices)
net = BacktestPortfolioTarget(fee=2e-4)(targets, prices)
for name, out in [("no fee", gross), ("2 bps fee", net)]:
    print(f"{name:9s}: {out[:, :, 0].sum(axis=1)[-1] * 1e4:+.1f} bps of capital, "
          f"{int(changed[61:].sum())} position changes")
```

```{code-cell} ipython3
fig, (ax0, ax1, ax2) = plt.subplots(3, 1, figsize=(9, 7), sharex=True)
ax0.plot(t, beta, color="steelblue", label="Kalman")
ax0.plot(t, tls, color="darkorange", lw=0.9, label="TLS (120 min)")
ax0.set_ylabel("hedge ratio")
ax0.legend(fontsize=8, loc="lower right")
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

The spread does revert: the position earns a little before fees. Each position
change trades both legs, so the fee is paid on about the gross exposure every
time, and on one-minute bars that cost is several times the edge.

## Is it more than luck?

`position_test` with a portfolio engine shifts both legs' targets together by a
random offset and reruns the book, keeping the trades and only moving them in
time. The p-value is the share of shifted books that do at least as well as
the real one before fees.

```{code-cell} ipython3
res = position_test(BacktestPortfolioTarget(), targets, prices, n=999, metric="total_pnl")
print(f"p = {res.p_value:.3f}, real {res.observed * 1e4:+.1f} bps, "
      f"median shifted {np.median(res.null) * 1e4:+.1f} bps")
```

Related pages: [`KalmanRegression`](../functions_fin/KalmanRegression.md),
[`RollingADF`](../functions_fin/RollingADF.md),
[`BandSignal`](../functions_signal/BandSignal.md),
[`BacktestPortfolioTarget`](../functions_fin/BacktestPortfolioTarget.md), and
the [portfolios and pairs guide](../portfolio_backtesting.md).
