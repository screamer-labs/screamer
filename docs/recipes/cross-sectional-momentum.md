---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# A cross-sectional momentum book

This recipe builds a dollar-neutral momentum book on 20 crypto assets: it scores
each asset by its past return, ranks the scores across assets, turns the ranks
into long and short weights, rebalances once a week with fees and slippage, and
tests whether the result beats randomly timed copies of the same book.

## Data

The file `binance_daily_20.csv` holds daily bars of 20 Binance spot pairs quoted
in USDT, from 2022-01-01 to 2026-09-30, downloaded from Binance's public daily
klines. It is in long format, one row per day and asset, with the columns
`date, symbol, open, high, low, close, volume, quote_volume`. Every asset has
all 1734 days.

The 20 assets were chosen in 2026 from coins that are still large today. Coins
that collapsed or were delisted during the period (LUNA in 2022, FTT in 2022)
are missing, so the sample is subject to survivorship bias. For a long/short
book this matters: the assets that fell hardest are the ones a momentum book
would have been short.

Pivoting on `symbol` gives one `(T, K)` array per field, time down the rows and
one asset per column.

```{code-cell} ipython3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from screamer import (LogReturn, Lag, CrossRank, CrossNormalize,
                      BacktestPortfolioOHLCTarget, PortfolioReport, position_test)

df = pd.read_csv("../notebooks/data/binance_daily_20.csv", parse_dates=["date"])
symbols = list(df["symbol"].unique())
wide = df.pivot(index="date", columns="symbol")
o, h, l, c = (wide[f][symbols].to_numpy() for f in ["open", "high", "low", "close"])
dates = wide.index
c.shape
```

## Score and weights

The score is the log return from 30 days ago to 3 days ago: `LogReturn(27)` is
the 27-day log return, and `Lag(3)` moves it 3 days back. Leaving out the last
few days is a common choice in momentum research, where very recent returns
tend to partly reverse. The score on day `t` uses closes up to day `t`.

[`CrossRank`](../functions_cross/CrossRank.md) replaces each day's scores by
their ranks across the 20 assets, so the weights depend on the order of the
assets and not on the size of their returns.
[`CrossNormalize`](../functions_cross/CrossNormalize.md) with `gross=1.0,
net=0.0` scales each row of ranks to weights whose absolute values sum to 1 and
whose sum is 0: long the upper half, short the lower half, half the capital on
each side.

The book rebalances on Sundays. On every other day the target is `NaN`, which
tells the engine to keep the current positions.

```{code-cell} ipython3
score = Lag(3)(LogReturn(27)(c))
weights = CrossNormalize(gross=1.0, net=0.0)(CrossRank()(score))
weights[dates.dayofweek != 6] = np.nan          # trade only on Sundays

last = np.flatnonzero(~np.isnan(weights[:, 0]))[-1]
print(dates[last].date())
pd.Series(weights[last], index=symbols).sort_values().round(3).to_frame("weight").T
```

## Backtest

[`BacktestPortfolioOHLCTarget`](../functions_fin/BacktestPortfolioOHLCTarget.md)
runs the 20 assets from one account. A target decided at Sunday's close
executes one bar later (`delay=1`, the default), on Monday, at Monday's
`(open + high + low + close) / 4` (`fill_at="ohlc4"`) as a proxy for trading
through the day. On a market that trades around the clock, Monday's open is
Sunday's close up to a tick, so filling there would assume the whole book
trades the moment the decision is made.
Each trade pays a 10 bps fee, Binance's base spot taker rate, and 2 bps of
slippage. A run without costs shows how much the costs take.

Spot accounts cannot hold short positions. A live version of this book would
hold its shorts in perpetual futures or on margin and pay funding or borrowing
costs, which this backtest leaves out.

```{code-cell} ipython3
engine = BacktestPortfolioOHLCTarget(fee=1e-3, slippage=2e-4, fill_at="ohlc4")
out = engine(weights, o, h, l, c)                # (T, K, 4): equity, pnl, position, cost
frictionless = BacktestPortfolioOHLCTarget(fill_at="ohlc4")(weights, o, h, l, c)

equity = 1.0 + out[:, :, 0].sum(axis=1)          # initial capital 1
equity_free = 1.0 + frictionless[:, :, 0].sum(axis=1)
print(f"final equity {equity[-1]:.3f} with costs, {equity_free[-1]:.3f} without")
print(f"peak equity {equity.max():.3f} on {dates[equity.argmax()].date()}")
```

## Report

[`PortfolioReport`](../functions_fin/PortfolioReport.md) reduces the
`(T, K, 4)` output to running portfolio columns `[drawdown, cum_cost,
turnover, trades, max_drawdown, sharpe]`. Drawdown and cost are in units of the
initial capital, and `trades` counts asset position changes. Its turnover
column adds up position changes in units of each asset, which mixes BTC with
DOGE here, so the traded value below comes from the cost column instead: each
trade pays 12 bps of its value.

```{code-cell} ipython3
report = PortfolioReport()(out)
drawdown, cum_cost, _, trades, max_dd, sharpe = report[-1]
years = len(dates) / 365
print(f"asset trades       {trades:.0f}")
print(f"total cost         {cum_cost:.3f} of initial capital")
print(f"traded value       about {cum_cost / 12e-4 / years:.0f} x initial capital per year")
print(f"max drawdown       {max_dd:.3f} of initial capital")
```

The plot shows the book's equity with and without costs, and its gross and net
exposure: the value of the positions as a multiple of equity, summed in
absolute value and with signs. Both are reset to 1 and 0 on every Monday and
drift with prices in between.

```{code-cell} ipython3
value = out[:, :, 2] * c                         # position value per asset
gross = np.abs(value).sum(axis=1) / equity
net = value.sum(axis=1) / equity

fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(9, 5.5), sharex=True,
                               gridspec_kw={"height_ratios": [2, 1]})
ax0.plot(dates, equity_free, color="0.6", ls="--", lw=1, label="no costs")
ax0.plot(dates, equity, color="steelblue", lw=1.2, label="10 bps fee + 2 bps slippage")
ax0.axhline(1, color="k", lw=0.5)
ax0.set_ylabel("equity (initial capital 1)")
ax0.legend(loc="upper left", fontsize=8)
ax1.plot(dates, gross, color="indigo", lw=0.8, label="gross")
ax1.plot(dates, net, color="darkorange", lw=0.8, label="net")
ax1.axhline(0, color="k", lw=0.5)
ax1.set_ylabel("exposure")
ax1.legend(loc="upper left", fontsize=8, ncol=2)
plt.tight_layout()
```

The book gains until the end of 2023, holds most of the gain through 2024,
loses it from 2025 on, and ends at 0.86 once costs are paid. Costs take about a
fifth of the initial capital over the period.

## Is it better than chance?

[`position_test`](../functions_fin/position_test.md#a-portfolio) with a
portfolio engine shifts the whole `(T, K)` target by the same random offset in
each randomized copy, so every copy holds the same book composition at other
times, and compares the book's Sharpe ratio with those copies.

```{code-cell} ipython3
res = position_test(engine, weights, o, h, l, c, n=999, periods_per_year=365)
print(f"annualized Sharpe {res.observed:+.2f}, p = {res.p_value:.3f}")
```

With a p-value of 0.44, the momentum timing does no better than random timing
of the same book on this sample. The lookback and the skip were fixed
before the run; trying several and keeping the best would make any p-value too
small.

Related pages: [Portfolios and pairs](../portfolio_backtesting.md),
[`CrossRank`](../functions_cross/CrossRank.md),
[`CrossNormalize`](../functions_cross/CrossNormalize.md),
[`CrossTopBottom`](../functions_cross/CrossTopBottom.md),
[`BacktestPortfolioOHLCTarget`](../functions_fin/BacktestPortfolioOHLCTarget.md),
[`PortfolioReport`](../functions_fin/PortfolioReport.md), and
[`position_test`](../functions_fin/position_test.md).
