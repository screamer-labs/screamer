---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# Size a position by volatility

Volatility targeting sets the position to a target volatility divided by a
running estimate of the asset's volatility, here with `EwStd`, and
`BacktestOHLCTarget(size="weight")` trades it as a fraction of equity.

```{code-cell} ipython3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from screamer import (EwMean, EwStd, LogReturn, Return, RollingStd, Sign, FillNa,
                      Clip, Round, BacktestOHLCTarget, backtest_report)

df = pd.read_csv("../../notebooks/data/binance_daily_20.csv", parse_dates=["date"])
btc = df[df["symbol"] == "BTC"].set_index("date")
o, h, l, c = (btc[k] for k in ["open", "high", "low", "close"])

direction = Sign()(EwMean(span=20)(c) - EwMean(span=100)(c))
vol = EwStd(span=60)(LogReturn()(c)) * np.sqrt(365)        # annual volatility
leverage = Round()(FillNa(0.0)(Clip(upper=2.0)(0.40 / vol)) * 10) / 10
weights = {"fixed, weight ±1": direction,
           "volatility target 40%": direction * leverage}

capital = 1_000_000
engine = BacktestOHLCTarget(taker_fee=1e-3, slippage=5e-4, size="weight",
                            initial_capital=capital)
realized, stats = pd.DataFrame(index=btc.index), {}
for name, w in weights.items():
    out = engine(w, o, h, l, c)
    realized[name] = RollingStd(90)(Return()(capital + out[:, 0])) * np.sqrt(365)
    stats[name] = backtest_report(out, periods_per_year=365)[1]
pd.DataFrame(stats).loc[["sharpe", "num_trades", "total_cost"]].round(2)
```

The direction is the moving-average trend rule of [Backtest a signal on OHLCV
bars](../backtest/ohlcv.md). The weight is `0.4 / vol`, capped at 2 (twice the
equity) by `Clip` and rounded to steps of 0.1 by `Round`, so that small changes
in the estimate do not each cause a trade. With `size="weight"` the engine
reads each target as a fraction of equity, `initial_capital` plus the
cumulative pnl, and trades when the target changes, paying a 0.1% taker fee
and 0.05% slippage on each trade. `realized` is the
annualized volatility of the account's daily returns over 90 days.

```{code-cell} ipython3
fig, ax = plt.subplots(figsize=(9, 3.5))
for name, color in zip(weights, ["0.5", "steelblue"]):
    ax.plot(realized.index, realized[name], color=color, label=name)
ax.axhline(0.40, color="k", lw=0.8, ls="--", label="target")
ax.set_ylabel("90-day realized vol (annual)")
ax.legend(loc="upper right", fontsize=8)
plt.tight_layout()
```

The fixed run's realized volatility follows BTC's own, from about 15% to 69% a
year; the targeted run stays between about 30% and 53%. To stay there it
trades 320 times instead of 34, and on this sample its Sharpe ratio is lower,
0.17 against 0.25.

## Go deeper

- [`EwStd`](../../functions_ew/EwStd.md), [Measure the average true range](../volatility/atr.md): two volatility estimates to divide by.
- [`BacktestOHLCTarget`](../../functions_fin/BacktestOHLCTarget.md): `size=`, `initial_capital`, and `rebalance_band` to limit trades.
- [Portfolios and pairs](../../portfolio_backtesting.md): sizing several assets from one account.
