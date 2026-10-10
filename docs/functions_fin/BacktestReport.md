---
name: BacktestReport
title: Running report columns for a backtest
implementation_family: fin
topics:
- backtesting
tags:
- backtest
- pnl
- drawdown
- turnover
- sharpe
- equity curve
- risk
- sharpe ratio
- max drawdown
- performance
- backtest statistics
short: Turn a backtest engine's [equity, pnl, position, cost] into running drawdown, cost,
  turnover, trades, and Sharpe.
inputs: 4
outputs: 6
parameters:
- name: periods_per_year
  type: float
  default: 1.0
  min: 0.0
  description: Steps per year, used to annualize the Sharpe ratio (for example 252 for daily
    bars). The default 1 gives the per-step ratio.
nan_policy: ignore
see_also:
- backtest_report
- BacktestStats
- BacktestPriceTarget
- Drawdown
- MaxDrawdown
output_names:
- drawdown
- cum_cost
- turnover
- trades
- max_drawdown
- sharpe
---

# `BacktestReport`

## Description

`BacktestReport` turns a backtest engine's per-step `[equity, pnl, position, cost]`
into the running report columns. The Python [`backtest_report`](backtest_report.md)
helper labels these columns and reads the last row for the summary.

The six outputs, in order, are:

0. `drawdown`: `equity - running peak equity`, the dollar drawdown (`<= 0`). This
   is the dollar form, not the percentage [`Drawdown`](Drawdown.md) op, because an
   equity curve starts at zero and can go negative, where a percentage of the peak
   is undefined.
1. `cum_cost`: the running sum of `cost`.
2. `turnover`: the running sum of `|position change|`; the first step counts the
   move from flat.
3. `trades`: the running count of steps that changed the position.
4. `max_drawdown`: the running minimum of `drawdown`, the worst so far (`<= 0`).
5. `sharpe`: the running `mean(pnl) / sample std(pnl)`, multiplied by
   `sqrt(periods_per_year)`, `NaN` until two finite `pnl` values with positive
   dispersion. With the default `periods_per_year=1` it is the per-step ratio.

[`BacktestStats`](BacktestStats.md) adds trade-level statistics (win rate, profit
factor, holding period), exposure, Sortino, and Calmar.

Every column is a causal accumulator, so a full-sample statistic is the last
finite value: `total_pnl` is the last equity, `max_drawdown` the last
`max_drawdown`, and so on.


<!-- NAN_FOOTNOTE_START -->
## NaN handling

**Policy: `ignore`.** A `NaN` in any input gives a `NaN` output at that step. The step is skipped: it does not enter the window or update the internal state. See [NaN and warmup](../nan_and_warmup.md).
<!-- NAN_FOOTNOTE_END -->

## Examples

### Usage plot

```{eval-rst}
.. plotly::
    :include-source: True

    import numpy as np
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from screamer import BacktestPriceTarget, BacktestReport

    rng = np.random.default_rng(0)
    n = 500
    price = 100 + np.cumsum(rng.standard_normal(n) * 0.3)
    signal = np.sign(rng.standard_normal(n))
    equity, pnl, position, cost = (BacktestPriceTarget(spread=0.0005)(signal, price).T)

    rep = BacktestReport()(equity, pnl, position, cost)   # (n, 6)
    drawdown, trades = rep[:, 0], rep[:, 3]

    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.4, 0.3, 0.3],
                        vertical_spacing=0.06)
    fig.add_trace(go.Scatter(y=equity, name='equity', line=dict(color='steelblue')), row=1, col=1)
    fig.add_trace(go.Scatter(y=drawdown, name='drawdown', line=dict(color='crimson'),
                             fill='tozeroy'), row=2, col=1)
    fig.add_trace(go.Scatter(y=trades, name='trades', line=dict(color='seagreen')), row=3, col=1)
    fig.update_layout(title='BacktestReport: running equity, drawdown, and trade count',
                      yaxis=dict(title='equity ($)'), yaxis2=dict(title='drawdown ($)'),
                      yaxis3=dict(title='trades'),
                      margin=dict(l=20, r=20, t=60, b=20),
                      legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1))
    fig.show()
```
