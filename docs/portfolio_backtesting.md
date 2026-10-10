# Portfolios and pairs

A portfolio strategy trades several assets from one account: it scores the
assets, turns the scores into weights, rebalances, and pays costs on every
asset it trades. A pairs trade is the smallest case, two assets held against
each other. screamer covers each step:

| Step | Functions |
|:-----|:----------|
| Compare assets at each step | [`CrossRank`](functions_cross/CrossRank.md), [`CrossZscore`](functions_cross/CrossZscore.md), [`CrossDemean`](functions_cross/CrossDemean.md), [`CrossWinsorize`](functions_cross/CrossWinsorize.md) |
| Turn scores into weights | [`CrossNormalize`](functions_cross/CrossNormalize.md), [`CrossTopBottom`](functions_cross/CrossTopBottom.md) |
| Backtest the book | [`BacktestPortfolioTarget`](functions_fin/BacktestPortfolioTarget.md), [`BacktestPortfolioOHLCTarget`](functions_fin/BacktestPortfolioOHLCTarget.md), [`PortfolioReport`](functions_fin/PortfolioReport.md) |
| Hedge ratio and spread of a pair | [`KalmanRegression`](functions_fin/KalmanRegression.md), [`RollingTLSBeta`](functions_fin/RollingTLSBeta.md), [`RollingBeta`](functions_fin/RollingBeta.md) |
| Is the pair still mean-reverting? | [`RollingADF`](functions_fin/RollingADF.md), [`RollingOU`](functions_rolling/RollingOU.md) |
| Entry and exit rules | [`BandSignal`](functions_signal/BandSignal.md) |
| Is the result luck? | [`position_test`](functions_fin/position_test.md#a-portfolio), [`strategy_test`](functions_fin/strategy_test.md) |

All of them take `(T, K)` arrays with time down the rows and one asset per
column, and all of them run one step at a time on a live feed with the same
results.

## From scores to weights

The cross-sectional operators work across the assets of one row. Each row is
processed on its own, so they are causal, and an asset that is `NaN` on a row
(not listed yet, halted) is left out of that row and gets `NaN` back.

A common chain ranks a score, so outliers and scale do not matter, and scales
the ranks to a dollar-neutral book with a gross exposure of 1:

```python
weights = CrossNormalize(gross=1.0, net=0.0)(CrossRank()(score))
```

`CrossZscore` keeps the distances between assets where ranks discard them;
`CrossWinsorize` first clips extreme scores to the row's quantiles. For a
long/short book with equal weights per side, use
`CrossNormalize()(CrossTopBottom(count=3)(score))`.

## The portfolio engines

[`BacktestPortfolioTarget`](functions_fin/BacktestPortfolioTarget.md) fills each
step's orders at that step's price; [`BacktestPortfolioOHLCTarget`](functions_fin/BacktestPortfolioOHLCTarget.md)
fills a target decided at a bar's close on a later bar, at the price `fill_at`
chooses. Both keep one account for all assets:

- **Targets** are weights of portfolio equity by default (`size="weight"`), or
  currency amounts (`"notional"`) or positions (`"units"`). Portfolio equity is
  `initial_capital` plus the summed pnl, so returns compound over the book.
- **A finite target is an order and `NaN` holds.** A rebalancing schedule is a
  target array with weights on the rebalance steps and `NaN` elsewhere.
- **Drift.** Between rebalances, positions drift from their weights as prices
  move. `rebalance_band` trades an asset back when it drifts by more than that
  share of equity; `rebalance="all"` then rebalances the whole book.
- **Limits.** `max_gross` and `max_net` scale down the orders of a step that
  would push the book past a gross or net exposure limit.
- **Costs** are per asset when you pass lists: `fee`, `slippage`,
  `tick_size`, `multiplier`, and `fee_per_contract`.
- **Missing prices.** An asset with a `NaN` price on a step does not trade or
  mark on that step and keeps its position, so the book's sum stays defined.

The output has shape `(T, K, 4)`, `[equity, pnl, position, cost]` per asset.
The book's equity is `initial_capital + out[:, :, 0].sum(axis=1)`.

The example compares three ways of holding the same equal-weight portfolio of
four assets with different volatilities: rebalancing every month, rebalancing
only when a weight drifts by more than 5 percentage points, and never
rebalancing.

```{eval-rst}
.. plotly::
    :include-source: True

    import numpy as np
    import plotly.graph_objects as go
    from screamer import BacktestPortfolioTarget

    rng = np.random.default_rng(5)
    T, K = 756, 4                                       # three years of daily prices
    vol = np.array([0.006, 0.01, 0.015, 0.02])
    prices = 100 * np.exp(np.cumsum(rng.standard_normal((T, K)) * vol + 2e-4, axis=0))

    monthly = np.full((T, K), np.nan)
    monthly[::21] = 0.25
    once = np.full((T, K), np.nan)
    once[0] = 0.25

    fig = go.Figure()
    for name, target, kw in [('monthly', monthly, {}),
                             ('drift band 5%', once, dict(rebalance_band=0.05)),
                             ('never', once, {})]:
        out = BacktestPortfolioTarget(fee=1e-3, **kw)(target, prices)
        equity = 1.0 + out[:, :, 0].sum(axis=1)
        weight = out[:, 3, 2] * prices[:, 3] / equity    # the most volatile asset's weight
        trades = int((np.diff(out[:, :, 2], axis=0) != 0).sum())
        fig.add_trace(go.Scatter(y=weight, mode='lines', name=f'{name} ({trades} trades)'))
    fig.add_hline(y=0.25, line_dash='dash', line_color='gray')
    fig.update_layout(title='Weight of the most volatile asset under three rebalancing rules',
                      xaxis_title='Day', yaxis_title='Weight',
                      margin=dict(l=20, r=20, t=80, b=20),
                      legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1))
    fig.show()
```

The band keeps the weight within 5 points of its target with far fewer trades
than a monthly schedule; without rebalancing the weight wanders with the
asset's price.

## A cross-sectional book

The next example ranks eight assets by their 20-day return each week, holds a
dollar-neutral book of the ranks, and executes at the next day's open with
fees and slippage. The returns are simulated with a little persistence, so
momentum has an edge here; with real data, the significance test below is
what tells you whether it does.

```{eval-rst}
.. plotly::
    :include-source: True

    import numpy as np
    import plotly.graph_objects as go
    from screamer import (BacktestPortfolioOHLCTarget, CrossNormalize, CrossRank,
                          PortfolioReport)

    rng = np.random.default_rng(8)
    T, K = 1000, 8
    r = np.zeros((T, K))
    for t in range(1, T):
        r[t] = 0.08 * r[t - 1] + rng.standard_normal(K) * 0.012
    close = 100 * np.exp(np.cumsum(r, axis=0))
    open_ = np.vstack([close[:1], close[:-1]]) * np.exp(rng.standard_normal((T, K)) * 0.002)
    high, low = np.maximum(open_, close) * 1.004, np.minimum(open_, close) * 0.996

    momentum = np.full((T, K), np.nan)
    momentum[20:] = np.log(close[20:] / close[:-20])
    weights = CrossNormalize(gross=1.0, net=0.0)(CrossRank()(momentum))
    weights[np.arange(T) % 5 != 4] = np.nan             # decide on Fridays, trade Monday

    engine = BacktestPortfolioOHLCTarget(fee=5e-4, slippage=3e-4)
    out = engine(weights, open_, high, low, close)
    equity = 1.0 + out[:, :, 0].sum(axis=1)
    report = PortfolioReport()(out)                     # drawdown, cost, turnover, trades, ...

    fig = go.Figure()
    fig.add_trace(go.Scatter(y=equity, mode='lines', name='equity'))
    fig.add_trace(go.Scatter(y=1.0 - report[:, 1], mode='lines', name='1 - cumulative cost',
                             line=dict(dash='dot')))
    fig.update_layout(title='Weekly dollar-neutral momentum on eight assets',
                      xaxis_title='Day', yaxis_title='Equity (initial capital 1)',
                      margin=dict(l=20, r=20, t=80, b=20),
                      legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1))
    fig.show()
```

## Pairs

A pairs trade holds one asset against another in the ratio that cancels their
common movement, the hedge ratio, and bets that the remaining spread returns to
its mean. Three choices make up the strategy:

1. **The hedge ratio.** [`RollingBeta`](functions_fin/RollingBeta.md) is the
   ordinary least squares slope over a window.
   [`RollingTLSBeta`](functions_fin/RollingTLSBeta.md) is the total least
   squares slope, which gives the same ratio whichever leg you call dependent.
   [`KalmanRegression`](functions_fin/KalmanRegression.md) lets the intercept
   and slope drift and updates them every step, and returns the spread and its
   predicted deviation directly.
2. **Whether to trade the pair at all.**
   [`RollingADF`](functions_fin/RollingADF.md) tests whether the spread still
   mean-reverts; [`RollingOU`](functions_rolling/RollingOU.md) measures how
   fast. A pair whose spread stopped reverting is a trend bet, not an
   arbitrage.
3. **Entry and exit.** [`BandSignal`](functions_signal/BandSignal.md) enters
   when the spread's z-score passes an entry level, exits inside an exit
   level, and can stop out a spread that keeps diverging.

The example below estimates the hedge ratio with a slow Kalman filter, so it
stays steady: with log prices near 4, a small error in the hedge ratio
moves the spread by four times as much. It standardizes the spread with
[`RollingZscore`](functions_rolling/RollingZscore.md) and trades only while
the rolling ADF statistic is below its 5% critical value.

The legs then go to a portfolio engine. With the spread on log prices, one
unit of spread is long one unit of value in the first leg and short the hedge
ratio times that value in the second; as weights of equity with a gross
exposure of 1, that is `1 / (1 + beta)` and `-beta / (1 + beta)`.

```{eval-rst}
.. plotly::
    :include-source: True

    import numpy as np
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from screamer import (BacktestPortfolioOHLCTarget, BandSignal, KalmanRegression,
                          RollingADF, RollingZscore)

    rng = np.random.default_rng(11)
    T = 1500
    common = np.cumsum(rng.standard_normal(T)) * 0.012
    spread = np.zeros(T)
    for t in range(1, T):
        spread[t] = 0.9 * spread[t - 1] + rng.standard_normal() * 0.004
    log_b = 4.0 + common
    log_a = 0.3 + 1.1 * log_b + spread
    close = np.exp(np.column_stack([log_a, log_b]))
    open_ = np.vstack([close[:1], close[:-1]])
    high, low = np.maximum(open_, close) * 1.002, np.minimum(open_, close) * 0.998

    kf = KalmanRegression(delta=1e-8, obs_var=1e-4)(log_a, log_b)
    beta, alpha = kf[:, 0], kf[:, 1]
    spread_hat = log_a - alpha - beta * log_b            # the estimated spread
    z = RollingZscore(60)(spread_hat)
    adf = RollingADF(window_size=250, lags=1)(spread_hat)
    position = BandSignal(entry=2.0, exit=0.0, stop=4.0)(z)
    position[np.nan_to_num(adf, nan=0.0) > -2.86] = 0.0  # trade only while the spread reverts
    position[:250] = 0.0                                 # warm-up

    legs = np.column_stack([position / (1 + beta), -position * beta / (1 + beta)])
    changed = np.r_[True, position[1:] != position[:-1]]
    legs[~changed] = np.nan                              # trade only when the position changes
    out = BacktestPortfolioOHLCTarget(fee=5e-4)(legs, open_, high, low, close)
    equity = 1.0 + out[:, :, 0].sum(axis=1)

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        subplot_titles=('spread z-score and position', 'equity'))
    fig.add_trace(go.Scatter(y=z, name='z-score', line=dict(color='gray')), row=1, col=1)
    fig.add_trace(go.Scatter(y=position, name='position', line=dict(shape='hv', color='crimson')),
                  row=1, col=1)
    fig.add_trace(go.Scatter(y=equity, name='equity'), row=2, col=1)
    fig.update_layout(title='A Kalman-filter pairs trade with a cointegration gate',
                      margin=dict(l=20, r=20, t=80, b=20), showlegend=False)
    fig.show()
```

The [pairs trade recipe](recipes/pairs-trade.md) runs the same steps on real
BTC and ETH prices.

## Testing a book

A good-looking book can still be luck, and a portfolio has more ways to be
lucky than one strategy. Both randomization tests work on a whole book:

- [`position_test`](functions_fin/position_test.md#a-portfolio) with a
  portfolio engine shifts all assets of the target together, keeping the
  book's composition, and scores the summed book.
- [`strategy_test`](functions_fin/strategy_test.md) accepts a strategy that
  returns a `(T, K, 4)` portfolio output, and re-runs the whole strategy,
  weights included, on randomized data.

```python
res = position_test(BacktestPortfolioOHLCTarget(fee=5e-4, slippage=3e-4),
                    weights, open_, high, low, close, n=999, periods_per_year=252)
res.p_value, res.observed_stats["sharpe"]
```

`res.null_stats` holds every statistic of every randomized book, for plots of
the null distribution next to the real result. See
[Backtesting best practices](backtesting_best_practices.md) for the other
checks a result needs before it is trusted.

## Live

The engines and operators keep their state between calls, so a live loop calls
them once per step with that step's row of `K` values:

```python
rank, weigh = CrossRank(), CrossNormalize(gross=1.0, net=0.0)
engine = BacktestPortfolioOHLCTarget(fee=5e-4)
for bar in feed:                                   # one row of K assets per bar
    target = weigh(rank(bar.score))
    row = engine(target, bar.open, bar.high, bar.low, bar.close)   # (K, 4)
```
