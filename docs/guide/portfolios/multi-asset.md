---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# Backtest a portfolio of many assets

`BacktestPortfolioOHLCTarget` trades `K` assets from one account: it takes one
`(T, K)` array per input, time down the rows and one asset per column, and
returns a `(T, K, 4)` array that `backtest_report` sums into portfolio
statistics.

```{code-cell} ipython3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from screamer import (LogReturn, EwStd, CrossNormalize,
                      BacktestPortfolioOHLCTarget, backtest_report, MaxDrawdown)

df = pd.read_csv("../../notebooks/data/binance_daily_20.csv",
                 parse_dates=["date"])
wide = df.pivot(index="date", columns="symbol")      # (T, K) per field
o, h, l, c = (wide[f].to_numpy() for f in ["open", "high", "low", "close"])
dates = wide.index

vol = EwStd(span=60)(LogReturn()(c))                  # (T, K), one per asset
books = {"equal weight": np.full(c.shape, 1 / c.shape[1]),
         "inverse volatility": CrossNormalize(gross=1.0)(1 / vol)}
engine = BacktestPortfolioOHLCTarget(fee=1e-3, slippage=5e-4)
out, stats = {}, {}
for name, weights in books.items():
    weights[:60] = np.nan                             # estimate warming up
    weights[dates.day != 1] = np.nan                  # rebalance monthly
    out[name] = engine(weights, o, h, l, c)           # (T, K, 4)
    running, summary = backtest_report(out[name], periods_per_year=365)
    dd = MaxDrawdown()(1 + running["equity"])[-1]
    stats[name] = {"sharpe": summary["sharpe"],
                   "max_drawdown_pct": 100 * dd, "num_trades": summary["num_trades"]}
```

```{code-cell} ipython3
fig, ax = plt.subplots(figsize=(9, 3.5))
for (name, res), color in zip(out.items(), ["0.5", "steelblue"]):
    ax.plot(dates, 1 + res[:, :, 0].sum(axis=1), color=color, label=name)
ax.axhline(1, color="k", lw=0.5)
ax.set_ylabel("equity (starting capital 1)")
ax.legend(loc="upper left", fontsize=8)
plt.tight_layout()
```

The data is daily bars of 20 Binance spot pairs quoted in USDT, from January
2022 to September 2026; pivoting on `symbol` gives the `(T, K)` arrays. A
single-input function such as `LogReturn` or `EwStd` treats each column of a
`(T, K)` array as its own series, so `vol` holds a running volatility for each
asset. The equal-weight book puts 1/20 of equity in each asset; the
inverse-volatility book weights each asset by `1 / vol`, scaled by
`CrossNormalize` so the weights sum to 1, which gives calmer assets more
capital. Both rebalance on the first day of each month and hold in between,
because a NaN target keeps the current position. Targets trade at the next
day's open, and every trade pays a 0.1% fee and 0.05% slippage.

```{code-cell} ipython3
pd.DataFrame(stats).round(2)
```

`backtest_report` reads the `(T, K, 4)` output as one book: it sums equity,
pnl and cost over the assets, `periods_per_year=365` annualizes the Sharpe
ratio of the daily pnl, and `num_trades` counts asset position changes. Its
`summary["per_asset"]` holds the same statistics for each asset alone. The
20 coins were chosen in 2026 from coins that are still large, so the sample
leaves out coins that collapsed during the period. Both books lost more than
half their capital in 2022; over the whole sample the inverse-volatility book
did slightly better.

## Go deeper

- [Rank assets by momentum](../strategies/cross-sectional-momentum.md): a long-short book from ranked scores.
- [`BacktestPortfolioOHLCTarget`](../../functions_fin/BacktestPortfolioOHLCTarget.md): sizing, `rebalance_band`, and gross and net caps.
- [`backtest_report`](../../functions_fin/backtest_report.md): the book and per-asset statistics.
- [`PortfolioReport`](../../functions_fin/PortfolioReport.md): the running report as a node, and turnover in money with a price array.
- [Polymorphic input/output behavior](../../polymorphic_api.md): the shape rules for `(T, K)` inputs to every function.
- [Portfolios and pairs](../../portfolio_backtesting.md): the portfolio engines in depth.
