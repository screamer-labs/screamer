---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# Target a constant volatility

A volatility-targeting rule holds a long position whose size is a target
volatility divided by a running estimate of the asset's volatility, here
`EwStd` of daily log returns, so the position shrinks when the market is
turbulent and grows when it is calm.

```{code-cell} ipython3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from screamer import (EwStd, LogReturn, Clip, FillNa, Round,
                      BacktestOHLCTarget, backtest_report, MaxDrawdown)

df = pd.read_csv("../../notebooks/data/binance_daily_20.csv",
                 parse_dates=["date"])
btc = df[df["symbol"] == "BTC"].set_index("date")
o, h, l, c = (btc[k] for k in ["open", "high", "low", "close"])

vol = EwStd(span=60)(LogReturn()(c)) * np.sqrt(365)       # annual volatility
weight = Round()(FillNa(0.0)(Clip(upper=2.0)(0.40 / vol)) * 10) / 10
weight[:60] = 0.0                           # estimate still warming up

engine = BacktestOHLCTarget(taker_fee=1e-3, slippage=5e-4,
                            size="weight", initial_capital=1.0)
out = engine(weight, o, h, l, c)
hold = engine(np.sign(weight), o, h, l, c)        # weight 1 from day 60
running, summary = backtest_report(out, periods_per_year=365)
```

The target is 40% a year, a little below BTC's average volatility of about
50% on this sample. The weight is `0.4 / vol` times equity, capped at 2 by
`Clip` and rounded to steps of 0.1 by `Round`, so that small changes in the
estimate do not each cause a trade. Both runs stay flat for the first 60
days, while the estimate warms up; after that `hold` keeps a weight of 1, as a
reference. The data is daily Binance BTC/USDT bars from January 2022 to
September 2026; each target trades at the next day's open, and every fill pays
a 0.1% taker fee and 0.05% slippage.

```{code-cell} ipython3
fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(9, 5), sharex=True,
                               gridspec_kw={"height_ratios": [1, 2]})
ax0.step(btc.index, weight, where="post", color="indigo", lw=0.8)
ax0.set_ylabel("weight")
ax1.plot(btc.index, 1 + out[:, 0], color="steelblue", label="40% target")
ax1.plot(btc.index, 1 + hold[:, 0], color="0.6", lw=0.8, label="buy and hold")
ax1.axhline(1, color="k", lw=0.5)
ax1.set_ylabel("equity (starting capital 1)")
ax1.legend(loc="upper left", fontsize=8)
plt.tight_layout()
```

```{code-cell} ipython3
dd = MaxDrawdown()(1 + out[:, 0])[-1]           # fraction of the peak
pd.Series({"sharpe": summary["sharpe"], "max_drawdown_pct": 100 * dd,
           "num_trades": summary["num_trades"]}).round(2)
```

On this sample the targeted position has the same Sharpe ratio as holding one
times equity, 0.28, and a shallower worst drawdown: 59% of its peak equity,
against 67% for holding.

## Go deeper

- [Size a position by volatility](../signal-to-position/volatility-sizing.md): the same sizing applied to a trend rule, with the realized volatility it gives.
- [`EwStd`](../../functions_ew/EwStd.md), [Measure the average true range](../volatility/atr.md): two volatility estimates to divide by.
- [`BacktestOHLCTarget`](../../functions_fin/BacktestOHLCTarget.md): `size="weight"` and `rebalance_band`.
