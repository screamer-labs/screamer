---
name: BacktestL1
title: Backtest a two-sided market maker against L1 quotes
implementation_family: fin
topics:
- risk
tags:
- backtest
- pnl
- mark to market
- equity curve
- market making
- l1
- top of book
- limit order
- inventory
- transaction cost
- risk
short: "Backtest a two-sided market maker against a top-of-book (L1) quote stream, into a costed equity curve."
inputs: 8
outputs: 4
parameters:
- name: maker_fee
  type: float
  default: 0.0
  description: Fractional fee on a fill; negative for a maker rebate.
- name: fill
  type: str
  default: touch
  description: Fill rule. "touch" fills when the opposite side of the market reaches your quote, "breach" only when it trades through.
- name: max_position
  type: float
  default: .inf
  description: Inventory ceiling; buy fills are capped so the position never exceeds it.
- name: min_position
  type: float
  default: -.inf
  description: Inventory floor; sell fills are capped so the position never falls below it.
nan_policy: ignore
see_also:
- BacktestTrades
- BacktestOHLC
- BacktestSignal
- backtest_report
---

# `BacktestL1`

## Description

`BacktestL1` is a two-sided market-making backtest against a top-of-book (L1)
stream. Each event carries the market quote `(bid, ask, bid_size, ask_size)` and
the strategy's own quote `(my_bid, my_bid_size, my_ask, my_ask_size)` (align your
quote stream to the market with `combine_latest` upstream). Both sides rest, and
either or both can be lifted on one event, so the position is whatever the market
fills.

The resting buy at `my_bid` fills when the market can sell into it, the market
`ask` reaching `my_bid` (`ask <= my_bid` for `"touch"`, strict for `"breach"`), for
`min(my_bid_size, ask_size)` at `my_bid`; the resting sell fills symmetrically when
`bid >= my_ask`. Both are maker fills at the quoted price paying `maker_fee`
(negative for a rebate); the favorable fill versus the mid is the captured spread,
so a passive quote earns the half-spread while an aggressive one pays it. Fills are
capped so the position stays within `[min_position, max_position]` (unbounded by
default), full up to the available size (no queue-position modelling). Positions
mark to the mid.

Inputs are `(bid, ask, bid_size, ask_size, my_bid, my_bid_size, my_ask,
my_ask_size)`. It emits the four positional columns shared by the backtest family:
`0 = equity` (cumulative dollar PnL), `1 = pnl` (per event), `2 = position`, and
`3 = cost` (per event). A `NaN` in the market quote skips the event
(`nan_policy: ignore`); a `NaN` own-quote price means that side is not quoted.
[`backtest_report`](backtest_report.md) summarizes the resulting equity curve.

## Examples

### Usage plot

```{eval-rst}
.. plotly::
    :include-source: True

    import numpy as np
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from screamer import BacktestL1

    rng = np.random.default_rng(3)
    n = 800
    mid = 100 + np.cumsum(rng.standard_normal(n) * 0.02)
    half = 0.05
    bid, ask = mid - half, mid + half
    bid_size = ask_size = np.full(n, 5.0)

    # quote one tick inside the market on both sides, size 1, inventory bounded to +/-10
    my_bid, my_ask = bid + 0.01, ask - 0.01
    one = np.ones(n)
    out = BacktestL1(maker_fee=-0.0001, max_position=10.0, min_position=-10.0)(
        bid, ask, bid_size, ask_size, my_bid, one, my_ask, one)
    eq, pos = out[:, 0], out[:, 2]

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.6, 0.4],
                        vertical_spacing=0.08)
    fig.add_trace(go.Scatter(y=eq, name='equity', line=dict(color='steelblue')), row=1, col=1)
    fig.add_trace(go.Scatter(y=pos, name='inventory', line=dict(color='mediumpurple')), row=2, col=1)
    fig.update_layout(title='BacktestL1: a two-sided maker capturing spread under an inventory cap',
                      yaxis=dict(title='equity ($)'), yaxis2=dict(title='inventory'),
                      margin=dict(l=20, r=20, t=60, b=20),
                      legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1))
    fig.show()
```
