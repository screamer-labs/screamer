---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# A pairs trade

A pairs trade holds one asset against another in the ratio that cancels their
common movement, and bets that the remaining spread returns to its average.
This recipe estimates the hedge ratio of ETH against BTC with `RollingBeta`,
forms the spread with `RollingSpread`, scales it to a z-score, and backtests
both legs in one `(T, 2)` call.

## Prices and spread

The spread is taken on log prices, so the hedge ratio is a ratio of percentage
moves. `RollingBeta(60)(x, y)` is the slope of `x` on `y` over the last 60
minutes, and `RollingSpread(60)(x, y)` is `x - beta * y` with the same slope.
`RollingZscore` expresses the spread in standard deviations from its recent
mean.

```{code-cell} ipython3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from screamer import (Resample, CombineLatest, RollingBeta, RollingSpread,
                      RollingZscore, BacktestPriceTarget)

def closes(path):
    trades = pd.read_csv(path)
    return Resample(freq=60_000, agg="last")(
        trades["price"].to_numpy(), trades["timestamp"].to_numpy())

btc, btc_t = closes("../notebooks/data/deribit_btc_perp_6h.csv")
eth, eth_t = closes("../notebooks/data/deribit_eth_perp_6h.csv")
prices, ts = CombineLatest()(btc, eth, index=[btc_t, eth_t])
t = pd.to_datetime(ts, unit="ms")

log_btc, log_eth = np.log(prices[:, 0]), np.log(prices[:, 1])
beta = RollingBeta(60)(log_eth, log_btc)
spread = RollingSpread(60)(log_eth, log_btc)
z = RollingZscore(30)(spread)
```

## Positions

The position in the spread is short when the z-score is high and long when it
is low, scaled linearly and capped at one unit once the z-score reaches 2 in
either direction (the dashed lines in the plot). One unit of spread is $10,000 of
ETH against `beta` times $10,000 of BTC, so the two legs are converted to
contract counts by dividing by each price.

```{code-cell} ipython3
units = -np.clip(np.nan_to_num(z) / 2, -1, 1)          # spread position in [-1, 1]
notional = 10_000
eth_qty = units * notional / prices[:, 1]
btc_qty = -units * np.nan_to_num(beta) * notional / prices[:, 0]

target = np.column_stack([btc_qty, eth_qty])           # (T, 2), same column order as prices
out = BacktestPriceTarget(fee=2e-4)(target, prices)    # (T, 2, 4): one backtest per leg
equity = out[:, :, 0].sum(axis=1)                      # both legs together
free = BacktestPriceTarget()(target, prices)[:, :, 0].sum(axis=1)
print(f"final equity {equity[-1]:+.2f} USD with a 2 bps fee, {free[-1]:+.2f} USD without")
```

The position changes a little every minute, and each change pays the fee, so
on this short sample the fee is most of the loss. Trading only when the z-score
crosses a threshold would cut the turnover.

```{code-cell} ipython3
fig, (ax0, ax1, ax2) = plt.subplots(3, 1, figsize=(9, 6.5), sharex=True)
ax0.plot(t, beta, color="steelblue")
ax0.set_ylabel("hedge ratio")
ax1.plot(t, z, color="0.3", lw=0.9)
ax1.axhline(2, color="crimson", lw=0.6, ls="--")
ax1.axhline(-2, color="crimson", lw=0.6, ls="--")
ax1.set_ylabel("spread z-score")
ax2.plot(t, free, color="0.6", ls="--", label="no fee")
ax2.plot(t, equity, color="seagreen", label="2 bps fee")
ax2.legend(fontsize=8, loc="lower left")
ax2.axhline(0, color="k", lw=0.5)
ax2.set_ylabel("equity (USD)")
plt.tight_layout()
```

The [pairs notebook](../notebooks/18-pairs-and-mean-reversion.ipynb) adds
`RollingAlpha` and an Ornstein-Uhlenbeck fit with `RollingOU`, which measures
how fast the spread reverts.

Related pages: [`RollingBeta`](../functions_fin/RollingBeta.md),
[`RollingSpread`](../functions_fin/RollingSpread.md),
[`RollingZscore`](../functions_rolling/RollingZscore.md), and
[`BacktestPriceTarget`](../functions_fin/BacktestPriceTarget.md).
