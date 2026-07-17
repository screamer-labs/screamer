---
name: BacktestTrades
title: Backtest a resting limit order against the trade tape
implementation_family: fin
topics:
- risk
tags:
- backtest
- pnl
- mark to market
- equity curve
- trades
- tape
- limit order
- event driven
- transaction cost
- market making
- risk
short: "Backtest a resting limit order against the trade tape, filling on crossing prints, into a costed equity curve."
inputs: 4
outputs: 4
parameters:
- name: maker_fee
  type: float
  default: 0.0
  description: Fractional fee on a fill; negative for a maker rebate.
- name: fill
  type: str
  default: touch
  description: Fill rule. "touch" fills when a print reaches the order price, "breach" only when it trades through.
nan_policy: ignore
see_also:
- BacktestL1
- BacktestOHLC
- BacktestSignal
- backtest_report
---

# `BacktestTrades`

## Description

`BacktestTrades` is an event-driven backtest against the trade tape. Each event is
a print `(trade_price, trade_size)` together with the strategy's current resting
limit order `(order_price, order_size)` (align your order stream to the tape with
`combine_latest` upstream). `order_size` is signed: positive rests a buy, negative
a sell, and `NaN` or zero is no order.

A resting order fills when a print crosses it, a buy when
`trade_price <= order_price` and a sell when `trade_price >= order_price`
(`fill = "touch"`, strict for `"breach"`), filling `min(order size, trade size)` at
the order price and paying `maker_fee` (negative for a rebate). Fills are partial
up to the print size and front-of-queue (optimistic, no queue-position modelling).
Positions mark to the last trade price, so a fill just before the tape moves against
you shows up immediately as an adverse cost.

Inputs are `(order_price, order_size, trade_price, trade_size)`. It emits the four
positional columns shared by the backtest family: `0 = equity` (cumulative dollar
PnL), `1 = pnl` (per event), `2 = position`, and `3 = cost` (per event). A `NaN`
trade field skips the event (`nan_policy: ignore`); a `NaN` order price simply means
no resting order. [`backtest_report`](backtest_report.md) summarizes the resulting
equity curve.

## Examples

### Usage plot

```{eval-rst}
.. plotly::
    :include-source: True

    import numpy as np
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from screamer import BacktestTrades

    rng = np.random.default_rng(2)
    n = 600
    price = 100 + np.cumsum(rng.standard_normal(n) * 0.05)       # the trade tape
    size = np.abs(rng.standard_normal(n)) + 0.5

    # a passive buyer: always rest a bid one tick below the last print, size 1
    order_price = price - 0.02
    order_size = np.ones(n)

    out = BacktestTrades(maker_fee=-0.0001)(order_price, order_size, price, size)
    eq, pos = out[:, 0], out[:, 2]

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.6, 0.4],
                        vertical_spacing=0.08)
    fig.add_trace(go.Scatter(y=eq, name='equity', line=dict(color='steelblue')), row=1, col=1)
    fig.add_trace(go.Scatter(y=pos, name='inventory', line=dict(color='darkorange')), row=2, col=1)
    fig.update_layout(title='BacktestTrades: a passive bid accumulating inventory off the tape',
                      yaxis=dict(title='equity ($)'), yaxis2=dict(title='inventory'),
                      margin=dict(l=20, r=20, t=60, b=20),
                      legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1))
    fig.show()
```
