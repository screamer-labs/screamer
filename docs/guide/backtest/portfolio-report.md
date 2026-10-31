---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# Report a portfolio backtest as one book

`backtest_report` takes the `(T, K, 4)` output of a portfolio engine and
reports the `K` assets as one book, the account they share, with each asset's
own summary in `summary["per_asset"]`.

```{code-cell} ipython3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from screamer import (EwMean, Sign, Diff, BacktestPortfolioOHLCTarget,
                      backtest_report, MaxDrawdown)

df = pd.read_csv("../../notebooks/data/binance_daily_20.csv",
                 parse_dates=["date"])
wide = df.pivot(index="date", columns="symbol")       # (T, K) per field
o, h, l, c = (wide[f].to_numpy() for f in ["open", "high", "low", "close"])
direction = Sign()(EwMean(span=20)(c) - EwMean(span=100)(c))
direction[:100] = 0.0
weights = direction / c.shape[1]                      # 1/20 of equity each
weights[Diff()(direction) == 0] = np.nan              # trade on a change only

out = BacktestPortfolioOHLCTarget(fee=1e-3, slippage=5e-4)(weights, o, h, l, c)
running, summary = backtest_report(out, periods_per_year=365)
per_asset = pd.DataFrame(summary["per_asset"], index=wide["close"].columns)
print(f"book: pnl {summary['total_pnl']:.2f}, Sharpe {summary['sharpe']:.2f}, "
      f"max drawdown {100 * MaxDrawdown()(1 + running['equity'])[-1]:.0f}%")
per_asset[["total_pnl", "sharpe", "round_trips"]].sort_values("total_pnl").round(2)
```

```{code-cell} ipython3
fig, ax = plt.subplots(figsize=(9, 3.5))
ax.plot(wide.index, out[:, :, 0], color="0.75", lw=0.6)
ax.plot(wide.index, running["equity"], color="steelblue", label="book")
ax.axhline(0, color="k", lw=0.5)
ax.set_ylabel("profit (starting capital 1)")
ax.legend(loc="upper left", fontsize=8)
plt.tight_layout()
```

The book applies the 20/100-day trend rule of [Backtest a signal on OHLCV
bars](ohlcv.md) to each of 20 Binance coins, long or short 1/20 of equity in
each, from one account with a starting capital of 1. A NaN weight keeps the
position, so each asset trades only when its direction changes. Each trade
pays a 0.1% fee and 0.05% slippage at the next day's open.

`backtest_report` sums `equity`, `pnl` and `cost` over the assets into the
book, so `running` has those three columns and no `position`, since a book
has no single position. The Sharpe ratio, drawdown, Sortino and Calmar ratios
come from the summed book. `num_trades` and `turnover` add up the assets'
position changes, and the round-trip statistics, such as `win_rate`, pool the
round trips of every asset. Entry `k` of `summary["per_asset"]` is the
summary of `backtest_report(out[:, k, :])`, the asset alone, so the asset
pnls add up to the book's.

On a row where an asset has a NaN field, `backtest_report` skips that asset: it
adds no pnl, and its equity holds. An asset that starts trading later joins
the book on its first row. The gray lines in the plot are the 20 assets'
equity curves, and the blue one is their sum. The book's profit over the
period was 139% of its starting capital, with a Sharpe ratio of 0.42, and it
lost almost half of its peak equity at its worst.

## Go deeper

- [`backtest_report`](../../functions_fin/backtest_report.md): every statistic, the NaN rules for a book, and `by=` for results per month.
- [Backtest a portfolio of many assets](../portfolios/multi-asset.md): equal-weight and inverse-volatility books.
- [`PortfolioReport`](../../functions_fin/PortfolioReport.md): the running book report as a node, and turnover in money.
- [Portfolios and pairs](../../portfolio_backtesting.md): the portfolio engines in depth.
