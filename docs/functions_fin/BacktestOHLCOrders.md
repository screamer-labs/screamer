---
name: BacktestOHLCOrders
title: Backtest a two-sided order poster on OHLC bars
implementation_family: fin
topics:
- backtesting
tags:
- backtest
- pnl
- mark to market
- equity curve
- market making
- ohlc
- limit order
- inventory
- transaction cost
- risk
short: "Backtest a two-sided order poster on OHLC bars, posting resting bids and asks that fill when the bar's range reaches them."
inputs: 8
outputs: 4
parameters:
- name: maker_fee
  type: float
  default: 0.0
  description: Fractional fee on a passive (resting) fill; negative for a rebate.
- name: taker_fee
  type: float
  default: 0.0
  description: Fractional fee on a marketable (taker) fill.
- name: fill
  type: str
  default: touch
  description: Fill rule. "touch" fills when the bar's low/high reaches your quote price; "breach" requires the bar to trade through it.
- name: participation_ratio
  type: float
  default: 1.0
  min: 0.0
  max: 1.0
  description: Fraction of the quoted size captured on each fill (front-of-queue at 1.0).
- name: tick_size
  type: float
  default: 0.0
  min: 0.0
  description: Price increment added (subtracted) to the open for a market buy (sell) to model taker slippage.
- name: min_position
  type: float
  default: -.inf
  description: Inventory floor; sell fills are capped so the position never falls below it.
- name: max_position
  type: float
  default: .inf
  description: Inventory ceiling; buy fills are capped so the position never exceeds it.
nan_policy: ignore
see_also:
- BacktestOHLCTarget
- BacktestL1Orders
- BacktestTradesOrders
- backtest_report
---

# `BacktestOHLCOrders`

## Description

`BacktestOHLCOrders` runs a two-sided order-posting backtest on OHLC bars. Each
bar the strategy posts a resting bid `(bid_price, bid_size)` and a resting ask
`(ask_price, ask_size)`. Both sides execute on the same bar (same-bar fill model,
no deferral), so the position is whatever the bar's range allows.

**Passive fills.** A resting buy fills when the bar's low reaches the bid price.
In `"touch"` mode (default) the fill triggers when `low <= bid_price`; in
`"breach"` mode only when `low < bid_price` (the bar must trade strictly through).
The fill quantity is `min(bid_size, participation_ratio * bid_size, room)`, where
`room = max_position - current_position`. The fill price is the bid price and pays
`maker_fee`. The sell side mirrors on the high.

**Market fills.** A NaN bid or ask price is a market order (via `market_limit`):
the buy fills at `open + tick_size` and the sell fills at `open - tick_size`, both
paying `taker_fee`. An explicit `MARKET` constant (equal to `+inf`) also triggers
this path.

**Inventory bounds.** Fills are capped at each step so the position stays in
`[min_position, max_position]` (unbounded by default). Both sides may fill on the
same bar when the bar's range is wide enough.

Inputs are `(bid_price, bid_size, ask_price, ask_size, open, high, low, close)`.
Outputs are the four standard backtest columns: `0 = equity` (cumulative dollar
PnL), `1 = pnl` (per bar), `2 = position`, `3 = cost` (per bar). A `NaN` in any
of the bar fields (open/high/low/close) skips the bar and returns all-NaN
(`nan_policy: ignore`). A NaN `bid_price` or `ask_price` triggers a market order;
a NaN or zero `bid_size` / `ask_size` suppresses that side entirely.
[`backtest_report`](backtest_report.md) summarizes the resulting equity curve.

## Limitations

OHLC bars carry no intra-bar path, so the engine cannot distinguish whether the
low preceded the high or vice versa. When both the bid and ask fill on the same
bar the engine processes the buy first, then the sell; the ordering can affect the
position and PnL when inventory bounds are active. Use [`BacktestL1Orders`](BacktestL1Orders.md)
or [`BacktestL1TradesOrders`](BacktestL1TradesOrders.md) with a tick-level or quote-level feed
when intra-bar sequencing matters.

## Examples

### Usage plot

```{eval-rst}
.. plotly::
    :include-source: True

    import numpy as np
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from screamer import BacktestOHLCOrders

    rng = np.random.default_rng(7)
    n = 300
    t = np.arange(n)
    # synthetic mean-reverting close price
    close = 100 + 2 * np.sin(2 * np.pi * t / n * 6) + np.cumsum(rng.standard_normal(n) * 0.15)
    o = close - rng.uniform(0.05, 0.15, n)
    h = close + rng.uniform(0.2, 0.5, n)
    l = close - rng.uniform(0.2, 0.5, n)

    # post a bid 0.2 below close, ask 0.2 above; inventory bounded to +/-8
    bid = close - 0.2
    ask = close + 0.2
    one = np.ones(n)
    out = BacktestOHLCOrders(maker_fee=-0.0001, max_position=8.0, min_position=-8.0)(
        bid, one, ask, one, o, h, l, close)
    eq, pos = out[:, 0], out[:, 2]

    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.4, 0.3, 0.3],
                        vertical_spacing=0.06)
    fig.add_trace(go.Scatter(y=close, name='close', line=dict(color='gray')), row=1, col=1)
    fig.add_trace(go.Scatter(y=eq, name='equity', line=dict(color='steelblue')), row=2, col=1)
    fig.add_trace(go.Scatter(y=pos, name='inventory', line=dict(color='mediumpurple')), row=3, col=1)
    fig.update_layout(title='BacktestOHLCOrders: two-sided bar order poster, inventory capped at +/-8',
                      yaxis=dict(title='close'), yaxis2=dict(title='equity ($)'),
                      yaxis3=dict(title='inventory'),
                      margin=dict(l=20, r=20, t=60, b=20),
                      legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1))
    fig.show()
```
