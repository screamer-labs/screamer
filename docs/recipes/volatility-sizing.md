---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# Volatility-targeted position sizing

Volatility targeting sizes a position so that its expected risk stays near a
chosen level: the position is the target volatility divided by a running
estimate of the asset's volatility. This recipe applies it to a trend rule on
daily BTC bars, with a 40% annual volatility target, and compares it with a
position of plus or minus the full equity, sized at each entry.

## Weights

The direction comes from the trend rule of
[Your first strategy](first-strategy.md). `EwStd` of the daily log returns,
times the square root of 365, estimates the annual volatility; for BTC it ranges
from 27% to 94% over this sample. The weight is `0.4 / volatility`, capped at
2 (twice the equity) by `Clip` and rounded to steps of 0.1 so that small changes
in the estimate do not each cause a trade.

```{code-cell} ipython3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from screamer import (EwMean, EwStd, LogReturn, Return, RollingStd, Sign, FillNa,
                      Clip, Round, BacktestOHLCTarget, backtest_report)

bars = pd.read_csv("../notebooks/data/binance_daily_20.csv", parse_dates=["date"])
btc = bars[bars["symbol"] == "BTC"]
t = btc["date"].to_numpy()
o, h, l, c = (btc[k].to_numpy() for k in ["open", "high", "low", "close"])

direction = Sign()(FillNa(0.0)(EwMean(span=20)(c) - EwMean(span=100)(c)))
vol = EwStd(span=60)(LogReturn()(c)) * np.sqrt(365)
leverage = FillNa(0.0)(Clip(upper=2.0)(0.40 / vol))
leverage = Round()(leverage * 10) / 10

weights = {"fixed, weight ±1": direction, "volatility target 40%": direction * leverage}
```

## Backtest

With `size="weight"` the engine reads each target as a fraction of equity,
`initial_capital` plus the cumulative pnl, and converts it to BTC at the fill
price. It trades when the target changes, so the volatility-targeted run
rebalances each time the rounded leverage moves. `max_position` and
`min_position` would add a hard limit on the position in BTC; they bound units,
not weights.

```{code-cell} ipython3
capital = 1_000_000
engine = BacktestOHLCTarget(taker_fee=1e-3, slippage=5e-4,
                            size="weight", initial_capital=capital)
out = {name: engine(w, o, h, l, c) for name, w in weights.items()}

realized = {}
for name, res in out.items():
    daily = Return()(capital + res[:, 0])            # daily return of the account
    realized[name] = RollingStd(90)(daily) * np.sqrt(365)
    s = backtest_report(res, periods_per_year=365)[1]
    lo, hi = np.nanpercentile(realized[name], [5, 95])
    print(f"{name:22s} 90-day vol 5%-95%: {lo:.2f}-{hi:.2f}  Sharpe {s['sharpe']:.2f}  "
          f"trades {s['num_trades']:4.0f}  cost {s['total_cost']:8.0f}")
```

The fixed run's 90-day realized volatility moves with BTC's own, between 20%
and 62% of equity a year in 90% of the windows. The volatility-targeted run
stays between 34% and 49%. That stability costs turnover: it trades 320 times
instead of 34, and on this sample its Sharpe ratio is lower, 0.17 against 0.25.

```{code-cell} ipython3
fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
for (name, res), color in zip(out.items(), ["0.5", "steelblue"]):
    ax0.plot(t, realized[name], color=color, label=name)
    ax1.plot(t, capital + res[:, 0], color=color, label=name)
ax0.axhline(0.40, color="k", lw=0.8, ls="--", label="target")
ax0.set_ylabel("90-day realized vol (annual)")
ax0.legend(loc="upper right", fontsize=8)
ax1.set_ylabel("account equity (USD)")
plt.tight_layout()
```

Related pages: [`BacktestOHLCTarget`](../functions_fin/BacktestOHLCTarget.md),
[`EwStd`](../functions_ew/EwStd.md),
[`backtest_report`](../functions_fin/backtest_report.md), and the
[Portfolios and pairs](../portfolio_backtesting.md) guide.
