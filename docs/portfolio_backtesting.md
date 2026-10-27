---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
---

# Portfolios and pairs

A portfolio strategy trades several assets from one account: it scores the
assets, turns the scores into weights, rebalances, and pays costs on every
asset it trades. A pairs trade is the smallest case, two assets held against
each other. This page shows how screamer covers each step, on daily bars of 20
crypto assets.

| Step | Functions |
|:-----|:----------|
| Compare assets at each step | [`CrossRank`](functions_cross/CrossRank.md), [`CrossZscore`](functions_cross/CrossZscore.md), [`CrossDemean`](functions_cross/CrossDemean.md), [`CrossWinsorize`](functions_cross/CrossWinsorize.md) |
| Turn scores into weights | [`CrossNormalize`](functions_cross/CrossNormalize.md), [`CrossTopBottom`](functions_cross/CrossTopBottom.md) |
| Backtest the book | [`BacktestPortfolioTarget`](functions_fin/BacktestPortfolioTarget.md), [`BacktestPortfolioOHLCTarget`](functions_fin/BacktestPortfolioOHLCTarget.md), [`PortfolioReport`](functions_fin/PortfolioReport.md) |
| Hedge ratio and spread of a pair | [`KalmanRegression`](functions_fin/KalmanRegression.md), [`RollingTLSBeta`](functions_fin/RollingTLSBeta.md), [`RollingBeta`](functions_fin/RollingBeta.md) |
| Is the pair still mean-reverting? | [`RollingADF`](functions_fin/RollingADF.md), [`RollingOU`](functions_rolling/RollingOU.md) |
| Entry and exit rules | [`BandSignal`](functions_signal/BandSignal.md) |
| Pick a book out of sample | [`walk_forward`](functions_fin/walk_forward.md) |
| Is the result luck? | [`position_test`](functions_fin/position_test.md#a-portfolio), [`strategy_test`](functions_fin/strategy_test.md) |

All of them take `(T, K)` arrays with time down the rows and one asset per
column.

## The data

The examples use daily bars of 20 Binance spot pairs quoted in USDT, from
2022-01-01 to 2026-09-30 (`notebooks/data/binance_daily_20.csv`). The assets
were chosen in 2026 from coins that are still large, so coins that failed
during the period are missing and the results carry survivorship bias. The
file is in long format, one row per day and asset; pivoting on `symbol` gives
one `(T, K)` array per field.

```{code-cell} ipython3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from screamer import (LogReturn, RollingStd, CrossRank, CrossZscore, CrossWinsorize,
                      CrossNormalize, CrossTopBottom, BacktestPortfolioTarget,
                      BacktestPortfolioOHLCTarget, PortfolioReport, position_test)

df = pd.read_csv("notebooks/data/binance_daily_20.csv", parse_dates=["date"])
symbols = list(df["symbol"].unique())
wide = df.pivot(index="date", columns="symbol")
o, h, l, c = (wide[f][symbols].to_numpy() for f in ["open", "high", "low", "close"])
dates = wide.index
T, K = c.shape
T, K
```

## From scores to weights

The cross-sectional operators work across the assets of one row. Each row is
processed on its own, so they are causal, and an asset that is `NaN` on a row
(not listed yet, halted) is left out of that row and gets `NaN` back.

The score here is each asset's 30-day log return. Three ways to turn it into
weights:

- `CrossRank` then `CrossNormalize(gross=1.0, net=0.0)`: the ranks, scaled to
  a dollar-neutral book whose absolute weights sum to 1. Outliers and the scale
  of the scores do not matter.
- `CrossWinsorize`, `CrossZscore`, then `CrossNormalize`: the z-scores keep the
  distances between assets that ranks discard. Winsorizing first clips extreme
  scores to the row's quantiles (here the 10% and 90% quantiles) so one asset
  cannot take most of the book.
- `CrossTopBottom(count=4)` then `CrossNormalize`: long the 4 highest scores and
  short the 4 lowest, with equal weights per side.

```{code-cell} ipython3
score = LogReturn(30)(c)
book = CrossNormalize(gross=1.0, net=0.0)
weights = {
    "rank": book(CrossRank()(score)),
    "z-score": book(CrossZscore()(CrossWinsorize(lower=0.1, upper=0.9)(score))),
    "top/bottom 4": book(CrossTopBottom(count=4)(score)),
}
pd.DataFrame({k: w[-1] for k, w in weights.items()}, index=symbols).round(3).T
```

The [cross-sectional momentum page](guide/strategies/cross-sectional-momentum.md)
backtests the rank-weighted book.

## The portfolio engines

[`BacktestPortfolioTarget`](functions_fin/BacktestPortfolioTarget.md) fills
each step's orders at that step's price, so a target must be decided before
that price is known.
[`BacktestPortfolioOHLCTarget`](functions_fin/BacktestPortfolioOHLCTarget.md)
fills a target decided at a bar's close on a later bar. Both keep one account
for all assets:

- **Targets** are weights of portfolio equity by default (`size="weight"`), or
  currency amounts (`"notional"`) or positions (`"units"`). Portfolio equity is
  `initial_capital` plus the summed pnl, so returns compound over the book.
- **A finite target is an order and `NaN` holds.** A rebalancing schedule is a
  target array with weights on the rebalance steps and `NaN` elsewhere.
- **Drift.** Between rebalances, positions drift from their weights as prices
  move. `rebalance_band` trades an asset back to its last target when its
  holding drifts by more than that share of equity; with `rebalance="all"`,
  one asset beyond the band rebalances the whole book.
- **Limits.** `max_gross` and `max_net` scale down the orders of a step that
  would push the book past a gross or net exposure limit, as multiples of
  equity.
- **Costs** are per asset when you pass lists: `fee`, `slippage`,
  `tick_size`, `multiplier`, and `fee_per_contract`.
- **Missing prices.** An asset with a `NaN` price on a step does not trade or
  mark on that step and keeps its position, so the book's sum stays defined.

The output has shape `(T, K, 4)`, `[equity, pnl, position, cost]` per asset,
where `equity` is the asset's cumulative pnl. The book's equity is
`initial_capital + out[:, :, 0].sum(axis=1)`.

### Rebalancing rules

The example holds the 20 assets in equal weights from the first day, with a
10 bps fee, under four rules: rebalance on the first day of every month, trade
an asset back when its weight drifts more than 2 percentage points from 5%, the
same band with `rebalance="all"`, and never rebalance. The equal weights use
no market data, so filling them at the same day's close is causal.

```{code-cell} ipython3
monthly = np.full((T, K), np.nan)
monthly[dates.is_month_start] = 1 / K
once = np.full((T, K), np.nan)
once[0] = 1 / K

rules = {
    "monthly": (monthly, {}),
    "band 2%": (once, dict(rebalance_band=0.02)),
    "band 2%, all": (once, dict(rebalance_band=0.02, rebalance="all")),
    "never": (once, {}),
}
runs, rows = {}, {}
for name, (target, kw) in rules.items():
    out = BacktestPortfolioTarget(fee=1e-3, **kw)(target, c)
    report = PortfolioReport()(out)
    runs[name] = out
    rows[name] = {"final equity": 1.0 + out[-1, :, 0].sum(),
                  "asset trades": report[-1, 3],
                  "total cost": report[-1, 1]}
pd.DataFrame(rows).T.round(3)
```

The band trades far less often than the monthly schedule. With
`rebalance="all"` every trip of the band rebalances all 20 assets, which costs
more trades than the monthly schedule. Without rebalancing, TRX grows from 5%
to over a third of the book:

```{code-cell} ipython3
trx = symbols.index("TRX")
fig, ax = plt.subplots(figsize=(9, 3.5))
for name in ["monthly", "band 2%", "never"]:
    out = runs[name]
    equity = 1.0 + out[:, :, 0].sum(axis=1)
    ax.plot(dates, out[:, trx, 2] * c[:, trx] / equity, lw=1, label=name)
ax.axhline(1 / K, color="k", lw=0.5, ls="--")
ax.set_ylabel("TRX weight")
ax.legend(loc="upper left", fontsize=8)
plt.tight_layout()
```

### Bars, delay, and per-asset costs

`BacktestPortfolioOHLCTarget` takes `(T, K)` arrays of target, open, high, low,
and close. A target decided at the close of bar `t` executes `delay` bars later
(default 1) at that bar's `fill_at` price: `"open"` (the default), `"close"`,
`"typical"`, or `"ohlc4"`. On a market that trades around the clock, the next
day's open is the decision close up to a tick, so filling there assumes the
book trades the moment the decision is made. The example fills at the next
day's `(open + high + low + close) / 4` instead, as a proxy for trading through
the day.

The book weights each asset by the inverse of its 30-day volatility,
long-only with gross exposure 1 (`CrossNormalize` with the default `net=None`
keeps the signs), rebalanced on Sundays. Slippage is 1 bp for BTC and ETH and
5 bps for the others, passed as a list with one value per asset.

```{code-cell} ipython3
vol = RollingStd(30)(LogReturn()(c))
inverse_vol = CrossNormalize(gross=1.0)(1.0 / vol)
inverse_vol[dates.dayofweek != 6] = np.nan       # decide on Sundays

slippage = [1e-4 if s in ("BTC", "ETH") else 5e-4 for s in symbols]
engine = BacktestPortfolioOHLCTarget(fee=1e-3, slippage=slippage, fill_at="ohlc4")
out = engine(inverse_vol, o, h, l, c)

equal = np.where(np.isnan(inverse_vol), np.nan, 1 / K)   # same schedule, equal weights
out_equal = engine(equal, o, h, l, c)                    # each call starts from a reset
```

[`PortfolioReport`](functions_fin/PortfolioReport.md) reduces the output to
running columns `[drawdown, cum_cost, turnover, trades, max_drawdown, sharpe]`
for the whole book, with drawdown and cost in units of the initial capital.
The engine output holds positions in units of each asset, and one unit of BTC
is worth far more than one unit of DOGE. Passing a `(T, K)` price array as a
second argument makes turnover the traded notional, the sum of
`abs(position change) * price` over the assets, in the same units as cost.
The example passes the fill price.

```{code-cell} ipython3
fill = (o + h + l + c) / 4
summary = {}
for name, run in [("inverse volatility", out), ("equal weight", out_equal)]:
    report = PortfolioReport()(run, fill)
    summary[name] = {"final equity": 1.0 + run[-1, :, 0].sum(),
                     "max drawdown": report[-1, 4],
                     "total cost": report[-1, 1],
                     "turnover": report[-1, 2]}
pd.DataFrame(summary).T.round(3)
```

## Testing a book

A good-looking book can still be luck, and a portfolio has more ways to be
lucky than one strategy.
[`position_test`](functions_fin/position_test.md#a-portfolio) with a portfolio
engine shifts all assets of the target together by one random offset per
copy, keeping the book's composition, and scores the summed book. Here it asks
whether the inverse-volatility weights did better at the times they were held
than at other times:

```{code-cell} ipython3
res = position_test(engine, inverse_vol, o, h, l, c, n=999, periods_per_year=365)
print(f"annualized Sharpe {res.observed:+.2f}, p = {res.p_value:.3f}")
```

The p-value of 0.55 says the timing of the weights adds nothing here.
`res.null_stats` holds every statistic of every randomized book, for plots of
the null distribution next to the real result.

[`walk_forward`](functions_fin/walk_forward.md) selects between candidate
books. Pass the candidates on a last axis, `(T, K, C)`; each training window is
scored on the summed pnl of the book, and the chosen book trades the test
window that follows. Here the candidates are the rank-weighted book on three
lookbacks, re-chosen every 90 days on the year before:

```{code-cell} ipython3
from screamer import LogReturn, walk_forward

lookbacks = (10, 30, 90)
candidates = np.stack([book(CrossRank()(LogReturn(n)(c))) for n in lookbacks], axis=2)
wf = walk_forward(engine, candidates, o, h, l, c, train=365, test=90, periods_per_year=365)
print("lookback per fold:", [lookbacks[k] if k >= 0 else None for k in wf.choice])
print(f"out-of-sample annualized Sharpe {wf.stats['sharpe']:+.2f}")
```

[`strategy_test`](functions_fin/strategy_test.md) re-runs a whole strategy on
randomized data and accepts a strategy that returns a `(T, K, 4)` book. It
takes `(T, K)` price panels: each randomized copy permutes the time order of
all assets together, so the correlation between them is kept. See
[Backtesting best practices](backtesting_best_practices.md) for the other
checks a result needs before it is trusted.

## Pairs

A pairs trade holds one asset against another in the ratio that cancels their
common movement, the hedge ratio, and trades the spread that remains: a
hedge-ratio estimator (`KalmanRegression`, `RollingTLSBeta`, `RollingBeta`), a
check that the spread still mean-reverts (`RollingADF`, `RollingOU`), and
entry and exit bands (`BandSignal`). The two legs then go to a portfolio engine
as weights. The [pairs trade page](guide/strategies/pairs.md) builds one on
real BTC and ETH prices.

## Live

The engines and the cross-sectional operators keep their state between calls,
and called with one row of `K` values they process one step, so a live loop
calls them once per bar and gets the batch results. A per-asset time-series
operator such as `RollingStd` treats a `(T, K)` array as `K` separate batches
and a 1-D row as one series, so a live loop keeps one instance per asset. The
loop below replays the inverse-volatility book one day at a time:

```{code-cell} ipython3
returns = [LogReturn() for _ in symbols]         # one instance per asset
vols = [RollingStd(30) for _ in symbols]
weigh = CrossNormalize(gross=1.0)
live = BacktestPortfolioOHLCTarget(fee=1e-3, slippage=slippage, fill_at="ohlc4")

for t in range(T):                               # one row of K assets per day
    vol_t = np.array([v(r(x)) for v, r, x in zip(vols, returns, c[t])])
    target = weigh(1.0 / vol_t) if dates[t].dayofweek == 6 else np.full(K, np.nan)
    row = live(target, o[t], h[t], l[t], c[t])   # (K, 4)
print(f"equity on {dates[-1].date()}: {1.0 + row[:, 0].sum():.3f}")
```

[Move a strategy from research to a live loop](guide/live/research-to-live.md)
connects a loop like this to a data feed.
