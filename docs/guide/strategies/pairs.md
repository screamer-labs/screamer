---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# Trade a pair

A pairs trade holds one asset against another in the ratio that cancels their
common movement and bets that the remaining spread returns to its average:
`KalmanRegression` estimates the ratio, `RollingZscore` and `BandSignal` set
the position, `RollingADF` switches it off when the spread stops reverting, and
`BacktestPortfolioOHLCTarget` trades both legs from one account.

```{code-cell} ipython3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from screamer import (Resample, CombineLatest, Ffill, KalmanRegression,
                      RollingZscore, RollingADF, BandSignal,
                      BacktestPortfolioOHLCTarget, PortfolioReport, MaxDrawdown)

def bars(path):
    trades = pd.read_csv(path)
    return Resample(freq=60_000, agg="ohlc", fill="nan")(
        trades["price"].to_numpy(), trades["timestamp"].to_numpy())

btc, btc_t = bars("../../notebooks/data/deribit_btc_perp_6h.csv")
eth, eth_t = bars("../../notebooks/data/deribit_eth_perp_6h.csv")
cols, ts = CombineLatest()(*btc.T, *eth.T, index=[btc_t] * 4 + [eth_t] * 4)
c = Ffill()(cols[:, [3, 7]])                   # (T, 2) closes: BTC, ETH
o, h, l = (np.where(np.isnan(cols[:, [k, k + 4]]), c, cols[:, [k, k + 4]])
           for k in range(3))                  # flat bar on a quiet minute

log_btc, log_eth = np.log(c).T
kf = KalmanRegression(delta=1e-10, obs_var=1e-6)(log_eth, log_btc)
beta, z = kf[:, 0], RollingZscore(30)(kf[:, 2])
position = BandSignal(entry=2.0, exit=0.0, stop=3.5)(z)
position[~(RollingADF(window_size=60, lags=1)(kf[:, 2]) < -3.34)] = 0.0
target = np.column_stack([-position * beta, position]) / (1 + beta)[:, None]
target[np.r_[False, position[1:] == position[:-1]]] = np.nan  # NaN holds
out = BacktestPortfolioOHLCTarget(fee=2e-4)(target, o, h, l, c)
```

The data is six hours of BTC and ETH perpetual trades on Deribit, turned into
one-minute bars by `Resample` and aligned on time by `CombineLatest`.
`KalmanRegression` regresses log ETH on log BTC with a slowly drifting
intercept and slope: column 0 is the hedge ratio `beta` and column 2 the
spread. `BandSignal` goes short the spread at a z-score of 2, long at -2, and
flat at 0, with a stop at 3.5. The position is held only while the rolling ADF
statistic is below -3.34, the 5% Engle-Granger critical value, which also
keeps it flat while the 60-minute ADF window fills. One unit of
spread is long ETH and short `beta` times as much BTC by value, with a gross
weight of 1; targets are sent only when the position changes, so the legs are
not rebalanced every minute. Each trade pays a 2 bps fee and fills at the next
minute's open.

```{code-cell} ipython3
t = pd.to_datetime(ts, unit="ms")
fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(9, 5), sharex=True)
ax0.plot(t, z, color="0.5", lw=0.8)
ax0.step(t, 2 * position, color="crimson", lw=1.0, where="post")
ax0.set_ylabel("z-score, position")
ax1.plot(t, out[:, :, 0].sum(axis=1) * 1e4, color="seagreen")
ax1.set_ylabel("pnl (bps of capital)")
plt.tight_layout()
```

```{code-cell} ipython3
*_, trades, _, sharpe = PortfolioReport()(out)[-1]
dd = MaxDrawdown()(1 + out[:, :, 0].sum(axis=1))[-1]   # fraction of the peak
pd.Series(dict(sharpe=sharpe, max_drawdown_pct=100 * dd, num_trades=trades))
```

`PortfolioReport` is the portfolio version of `backtest_report`. Its Sharpe
ratio is per minute and left that way, because six hours are too short to
annualize. The fees are the
whole loss: before fees the book ends close to flat, and each of the twelve
position changes trades both legs.

## Go deeper

- [Hedge ratios and spread estimation](../../notebooks/18-pairs-and-mean-reversion.ipynb): the estimators compared.
- [Check that a backtest result is not luck](../not-luck/index.md): `position_test` shifts both legs' targets in time.
- [`KalmanRegression`](../../functions_fin/KalmanRegression.md), [`RollingADF`](../../functions_fin/RollingADF.md), [`BandSignal`](../../functions_signal/BandSignal.md): the three steps of the signal.
- [Portfolios and pairs](../../portfolio_backtesting.md): sizing the legs of a book.
