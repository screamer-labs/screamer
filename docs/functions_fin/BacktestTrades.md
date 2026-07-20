---
name: BacktestTrades
title: Backtest a resting limit order against the trade tape
implementation_family: fin
topics:
- backtesting
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
  description: Fill rule. "touch" fills a participation partial when a print reaches the order price, "breach" only fills when it trades through.
- name: participation_ratio
  type: float
  default: 1.0
  min: 0.0
  max: 1.0
  description: Fraction of the at-price trade volume captured (front-of-queue at 1.0). A trade through your price fills the full order.
- name: min_position
  type: float
  default: -.inf
  description: Lower bound on the position. A fill that would push the position below this limit is truncated to the available room.
- name: max_position
  type: float
  default: .inf
  description: Upper bound on the position. A fill that would push the position above this limit is truncated to the available room.
nan_policy: ignore
see_also:
- BacktestL1Trades
- BacktestL1
- BacktestOHLCTarget
- backtest_report
---

# `BacktestTrades`

## Description

`BacktestTrades` is an event-driven backtest against the trade tape. Each event is
a print `(trade_price, trade_size)` together with the strategy's current resting
limit order `(order_price, order_size)` (align your order stream to the tape with
`combine_latest` upstream). `order_size` is signed: positive rests a buy, negative
a sell, and `NaN` or zero is no order.

A resting order fills against a print by where the print lands relative to the
order price. A print **through** the order price (a buy when `trade_price <
order_price`, a sell when `trade_price > order_price`) sweeps the level and fills
the **full order**: the small print at the through-price is the residual after the
sweep, so it does not bound the fill. A print **at** the order price fills
`min(order size, participation_ratio * trade_size)`, front-of-queue, capturing a
share of the volume that traded there. `fill = "breach"` keeps only the through
case. Fills pay `maker_fee` (negative for a rebate). Positions mark to the last
trade price, so a fill just before the tape moves against you shows up immediately
as an adverse cost.

Inputs are `(order_price, order_size, trade_price, trade_size)`. It emits the four
positional columns shared by the backtest family: `0 = equity` (cumulative dollar
PnL), `1 = pnl` (per event), `2 = position`, and `3 = cost` (per event). A `NaN`
trade field skips the event (`nan_policy: ignore`); a `NaN` order price simply means
no resting order. [`backtest_report`](backtest_report.md) summarizes the resulting
equity curve.

`min_position` and `max_position` set a static inventory cap. When a fill would
push the position outside `[min_position, max_position]`, the fill is truncated to
the available room rather than rejected outright. With the default unbounded cap
(`-inf`, `+inf`) the behaviour is identical to having no cap.

## Limitations

Fills are front-of-queue and optimistic: a trade at your price fills
`min(remaining, participation_ratio * trade_size)`, a trade through it fills the
full order. There is no queue-position modelling. Orders are counterfactual (zero
volume, no market impact), so a fill never alters the tape it reads.

## Examples

### Usage plot

```{eval-rst}
.. plotly::
    :include-source: True

    import numpy as np
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from screamer import BacktestTrades

    rng = np.random.default_rng(1)
    n = 600
    t = np.arange(n)
    price = 100 + 2 * np.sin(2 * np.pi * t / n * 3) + rng.standard_normal(n) * 0.25   # tape
    size = np.ones(n)

    # a mean-reverting maker: rest at the print, buy when cheap (< 100), sell when rich.
    # each at-price fill captures participation_ratio of the print, so inventory cycles.
    order_price = price
    order_size = np.sign(100 - price)
    out = BacktestTrades(participation_ratio=0.1, maker_fee=-0.0001)(
        order_price, order_size, price, size)
    eq, pos = out[:, 0], out[:, 2]

    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.4, 0.3, 0.3],
                        vertical_spacing=0.06)
    fig.add_trace(go.Scatter(y=price, name='price (tape)', line=dict(color='gray')), row=1, col=1)
    fig.add_trace(go.Scatter(y=eq, name='equity', line=dict(color='steelblue')), row=2, col=1)
    fig.add_trace(go.Scatter(y=pos, name='inventory', line=dict(color='darkorange')), row=3, col=1)
    fig.update_layout(title='BacktestTrades: a mean-reverting maker filling off the tape',
                      yaxis=dict(title='price'), yaxis2=dict(title='equity ($)'),
                      yaxis3=dict(title='inventory'),
                      margin=dict(l=20, r=20, t=60, b=20),
                      legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1))
    fig.show()
```
