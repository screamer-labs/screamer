---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# Rank assets by momentum

A cross-sectional momentum book scores each asset by its past return, ranks the
scores across assets with `CrossRank`, turns the ranks into long and short
weights with `CrossNormalize`, and trades them once a week with
`BacktestPortfolioOHLCTarget`.

```{code-cell} ipython3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from screamer import (LogReturn, Lag, CrossRank, CrossNormalize,
                      BacktestPortfolioOHLCTarget, backtest_report, MaxDrawdown)

df = pd.read_csv("../../notebooks/data/binance_daily_20.csv",
                 parse_dates=["date"])
wide = df.pivot(index="date", columns="symbol")      # (T, K) per field
o, h, l, c = (wide[f].to_numpy() for f in ["open", "high", "low", "close"])
dates = wide.index

score = Lag(3)(LogReturn(27)(c))                      # 30 to 3 days ago
weights = CrossNormalize(gross=1.0, net=0.0)(CrossRank()(score))
weights[dates.dayofweek != 6] = np.nan                # trade only on Sundays

engine = BacktestPortfolioOHLCTarget(fee=1e-3, slippage=5e-4)
out = engine(weights, o, h, l, c)                     # (T, K, 4)
```

```{code-cell} ipython3
fig, ax = plt.subplots(figsize=(9, 3.5))
ax.plot(dates, 1 + out[:, :, 0].sum(axis=1), color="steelblue")
ax.axhline(1, color="k", lw=0.5)
ax.set_ylabel("equity (starting capital 1)")
plt.tight_layout()
```

The data is daily bars of 20 Binance spot pairs quoted in USDT, from January
2022 to September 2026. Pivoting on `symbol` gives one `(T, K)` array per
field, time down the rows and one asset per column. The score is the log return
from 30 days ago to 3 days ago: `LogReturn(27)` is the 27-day log return and
`Lag(3)` moves it 3 days back, leaving out the most recent days, which tend to
partly reverse. `CrossRank` replaces each day's scores by their ranks across
the 20 assets, and `CrossNormalize(gross=1.0, net=0.0)` scales the ranks to
weights whose absolute values sum to 1 and whose sum is 0: long the upper half,
short the lower half. On every day but Sunday the target is NaN, which keeps
the current positions. A target decided at Sunday's close trades at Monday's
open, and every trade pays a 0.1% fee and 0.05% slippage.

```{code-cell} ipython3
running, summary = backtest_report(out, periods_per_year=365)
dd = MaxDrawdown()(1 + running["equity"])[-1]          # fraction of the peak
pd.Series({"sharpe": summary["sharpe"], "max_drawdown_pct": 100 * dd,
           "num_trades": summary["num_trades"]}).round(2)
```

`backtest_report` sums the `(T, K, 4)` output over the assets into one book;
`periods_per_year=365` annualizes the Sharpe ratio of the daily pnl, and
`num_trades` counts asset position changes. The book gains until the end of
2023, then gives the gain back and ends below its starting capital. The 20
assets were chosen in 2026 from coins that are still large, so coins that collapsed during the period
(LUNA and FTT in 2022) are missing; those are the ones a momentum book would
have been short. A live version would also hold its shorts in perpetual futures
or on margin and pay funding or borrowing costs, which this backtest leaves
out.

## Go deeper

- [`CrossRank`](../../functions_cross/CrossRank.md), [`CrossNormalize`](../../functions_cross/CrossNormalize.md), [`CrossTopBottom`](../../functions_cross/CrossTopBottom.md): scores to weights.
- [Backtest a portfolio of many assets](../portfolios/multi-asset.md): the `(T, K)` inputs and `(T, K, 4)` output.
- [Check that a backtest result is not luck](../not-luck/index.md): compare the book with randomly timed copies of itself.
- [Portfolios and pairs](../../portfolio_backtesting.md): sizing, rebalancing and reporting a book.
