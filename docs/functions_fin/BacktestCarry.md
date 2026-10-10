---
name: BacktestCarry
title: Add funding or carry to a backtest
implementation_family: fin
topics:
- backtesting
tags:
- backtest
- funding
- carry
- financing
- borrow
- perpetual
short: "Charge perpetual funding, borrow, or financing on the position of any backtest engine's output."
inputs: 6
outputs: 4
parameters:
- name: multiplier
  type: float
  default: 1.0
  min: 0.0
  description: Contract multiplier, as on the engine that produced the output.
nan_policy: ignore
see_also:
- BacktestPriceTarget
- BacktestOHLCTarget
- backtest_report
---

# `BacktestCarry`

## Description

`BacktestCarry` adds a carry charge to the output of any backtest engine: the
funding of a perpetual future, the borrow fee of a short, or the financing of a
leveraged position. Its inputs are the engine's `[equity, pnl, position, cost]`,
the mark `price`, and the per-step carry `rate`, all on the same steps.

The position held into step `t` pays

`carry_t = position_{t-1} * price_t * rate_t * multiplier`

so a positive rate charges longs and pays shorts, as perpetual funding does, and
a negative rate reverses that. The output has the engine's four columns with the
carry moved from pnl into cost: `equity` minus the cumulative carry,
`pnl - carry_t`, `position` unchanged, and `cost + carry_t`. The result feeds
[`backtest_report`](backtest_report.md) like any engine output.

The rate is per step. For funding paid every eight hours on hourly bars, pass
the funding rate on the funding hours and 0 elsewhere; for an annual borrow fee,
divide it by the number of steps per year. A `NaN` in any input, the rate
included, skips the row (`nan_policy: ignore`), so pass 0 on steps without
carry.

## Examples

### Usage plot

```{eval-rst}
.. plotly::
    :include-source: True

    import numpy as np
    import plotly.graph_objects as go
    from screamer import BacktestCarry, BacktestPriceTarget

    rng = np.random.default_rng(2)
    n = 24 * 60                                      # 60 days of hourly bars
    price = 30000 * np.exp(np.cumsum(rng.standard_normal(n) * 0.003))
    position = np.where(np.arange(n) < n // 2, 1.0, -1.0)   # long, then short

    # 8-hourly funding of about 0.01%, higher while the market is crowded long
    rate = np.zeros(n)
    rate[::8] = 1e-4 + 2e-4 * (np.arange(n)[::8] < n // 2)

    out = BacktestPriceTarget()(position, price)
    funded = BacktestCarry()(*out.T, price, rate)

    fig = go.Figure()
    fig.add_trace(go.Scatter(y=out[:, 0], name='without funding', line=dict(color='lightsteelblue')))
    fig.add_trace(go.Scatter(y=funded[:, 0], name='with funding', line=dict(color='steelblue')))
    fig.update_layout(title='BacktestCarry: perpetual funding charged on the held position',
                      xaxis_title='hour', yaxis_title='equity ($)',
                      margin=dict(l=20, r=20, t=60, b=20),
                      legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1))
    fig.show()
```
