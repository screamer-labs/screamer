---
name: BacktestPriceTarget
title: Backtest a target position on a value series
implementation_family: fin
topics:
- backtesting
tags:
- backtest
- pnl
- mark to market
- equity curve
- transaction cost
- strategy
- risk
short: "Backtest a target position against a value series (price/mark) into a costed mark-to-market equity curve."
inputs: 2
outputs: 4
parameters:
- name: spread
  type: float
  default: 0.0
  min: 0.0
  description: Fractional bid-ask spread crossed on each trade (e.g. 0.0005 = 5 bps). Default 0 is frictionless.
- name: fee
  type: float
  default: 0.0
  description: Fractional taker fee charged on the traded notional.
- name: min_position
  type: float
  default: -.inf
  description: Lower bound on the target position. Signals below this value are clamped to it.
- name: max_position
  type: float
  default: .inf
  description: Upper bound on the target position. Signals above this value are clamped to it.
nan_policy: ignore
see_also:
- Drawdown
- MaxDrawdown
- RollingInfoRatio
---

# `BacktestPriceTarget`

## Description

`BacktestPriceTarget` turns a target position on a value series (a price or mark)
into a costed profit-and-loss curve by reaching that target through taker liquidity.
The `signal` is the target position in units (its sign is long, short, or flat, any
magnitude); `price` is the mark (mid). Each bar the position moves to the signal via
a market order that crosses half of the fractional `spread` (a buy fills at
`price * (1 + spread/2)`, a sell at `price * (1 - spread/2)`) and pays the
fractional `fee` on the traded notional. With the default `spread = fee = 0` the
backtest is frictionless.

A target outside `[min_position, max_position]` is clamped to the nearest boundary
before the order is computed; the default bounds are unbounded so the behavior is
unchanged when the cap is not set.

It emits four positional columns: `0 = equity` (cumulative dollar PnL), `1 = pnl`
(per bar), `2 = position`, and `3 = cost` (per bar). It is causal: the signal at
`t` enters PnL only at `t+1`, through the position it sets, so a future signal
never changes a past row. A `NaN` signal or price skips the bar (an all-`NaN` row,
the position held across the gap; `nan_policy: ignore`).

The equity curve feeds the existing risk operators directly (dollar drawdown is
`equity - CumMax(equity)`), and [`backtest_report`](backtest_report.md) bundles
the common statistics (drawdown, cost, turnover, trades, Sharpe) in one call.

## Examples

### Usage plot

```{eval-rst}
.. plotly::
    :include-source: True

    import numpy as np
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from screamer import BacktestPriceTarget, RollingMean

    rng = np.random.default_rng(0)
    n = 500
    price = 100 + np.cumsum(rng.standard_normal(n) * 0.3)
    # a trend signal: long when the fast average is above the slow one, else short
    fast = RollingMean(10)(price)
    slow = RollingMean(50)(price)
    signal = np.sign(np.nan_to_num(fast - slow))

    free = BacktestPriceTarget()(signal, price)[:, 0]                    # frictionless equity
    costed = BacktestPriceTarget(spread=0.001, fee=0.0002)(signal, price)
    eq = costed[:, 0]
    dd = eq - np.maximum.accumulate(eq)                            # running dollar drawdown

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.6, 0.4],
                        vertical_spacing=0.08)
    fig.add_trace(go.Scatter(y=free, name='equity (frictionless)',
                             line=dict(color='gray', dash='dot')), row=1, col=1)
    fig.add_trace(go.Scatter(y=eq, name='equity (with cost)',
                             line=dict(color='steelblue')), row=1, col=1)
    fig.add_trace(go.Scatter(y=dd, name='drawdown', line=dict(color='crimson'),
                             fill='tozeroy'), row=2, col=1)
    fig.update_layout(title='BacktestPriceTarget: a trend signal turned into a costed equity curve',
                      yaxis=dict(title='equity ($)'), yaxis2=dict(title='drawdown ($)'),
                      margin=dict(l=20, r=20, t=60, b=20),
                      legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1))
    fig.show()
```
