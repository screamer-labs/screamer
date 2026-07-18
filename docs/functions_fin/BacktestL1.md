---
name: BacktestL1
title: Backtest a two-sided market maker against L1 quotes
implementation_family: fin
topics:
- backtesting
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
  description: Fractional fee on a maker fill; negative for a rebate.
- name: taker_fee
  type: float
  default: 0.0
  description: Fractional fee on a taker fill (a quote submitted already crossing the spread).
- name: fill
  type: str
  default: breach
  description: Fill rule. "breach" (conservative) fills only when the market trades through your quote; "touch" (optimistic) also fills a participation partial once per lock episode.
- name: participation_ratio
  type: float
  default: 1.0
  min: 0.0
  max: 1.0
  description: Fraction of the locked ask/bid size captured on a touch fill (front-of-queue at 1.0).
- name: tick_size
  type: float
  default: 0.0
  min: 0.0
  description: Price step a marketable order walks for the size beyond the displayed quote.
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
- BacktestL1Trades
- BacktestTrades
- BacktestOHLC
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

Fills follow where the market moves relative to your quote. When the market
`ask` trades **through** `my_bid` (`ask < my_bid`), the resting buy is run over
and fills its **full remaining** at `my_bid`, a maker fill paying `maker_fee`. In
`"touch"` mode a **lock** (`ask == my_bid`) also fills once on entry, for
`min(remaining, participation_ratio * ask_size)` at `my_bid`; further size changes
at the same locked price are ignored (a static quote cannot be told from a cancel,
so it does not re-fill). The `"breach"` default keeps only the through case. A
quote that appears **already crossing** the spread (`my_bid >= ask` on submission)
is a taker: it fills its full size at the market, walking `tick_size` for the part
beyond the displayed quote, and pays `taker_fee`. The sell side is symmetric.
Fills are capped so the position stays within `[min_position, max_position]`
(unbounded by default). Positions mark to the mid.

Inputs are `(bid, ask, bid_size, ask_size, my_bid, my_bid_size, my_ask,
my_ask_size)`. It emits the four positional columns shared by the backtest family:
`0 = equity` (cumulative dollar PnL), `1 = pnl` (per event), `2 = position`, and
`3 = cost` (per event). A `NaN` in the market quote skips the event
(`nan_policy: ignore`); a `NaN` own-quote price means that side is not quoted.
[`backtest_report`](backtest_report.md) summarizes the resulting equity curve.

## Limitations

Quotes-only fills are a heuristic: a size change at a static price is fill,
cancel, and re-quote all at once, and cannot be told apart. The `"breach"` default
fills only when the market trades through your quote, so it can **under-fill**.
`"touch"` adds a participation partial once per lock episode and can **over-fill**
when a lock's size came from cancels rather than trades. Prefer
[`BacktestL1Trades`](BacktestL1Trades.md) when a trade feed is available: real
trades drive the fills and remove this ambiguity. Orders are counterfactual (zero
volume, no market impact); a marketable order fills beyond the displayed size only
under the `tick_size` slippage assumption.

## Examples

### Usage plot

```{eval-rst}
.. plotly::
    :include-source: True

    import numpy as np
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from screamer import BacktestL1, Lag

    rng = np.random.default_rng(4)
    n = 600
    t = np.arange(n)
    tick = 0.5
    # a discrete tick-grid, mean-reverting mid (real books are gridded, so locks occur)
    mid = np.round((100 + 3 * np.sin(2 * np.pi * t / n * 4) + rng.standard_normal(n) * 0.4) / tick) * tick
    bid, ask = mid - tick, mid + tick
    five, one = np.full(n, 5.0), np.ones(n)

    # rest last event's touch on both sides; the market trading back through fills it.
    # touch mode captures a participation share on each lock; inventory bounded.
    my_bid = np.nan_to_num(Lag(1)(bid), nan=bid[0])
    my_ask = np.nan_to_num(Lag(1)(ask), nan=ask[0])
    out = BacktestL1(fill="touch", maker_fee=-0.0001, participation_ratio=0.5,
                     max_position=8.0, min_position=-8.0)(
        bid, ask, five, five, my_bid, one, my_ask, one)
    eq, pos = out[:, 0], out[:, 2]

    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.4, 0.3, 0.3],
                        vertical_spacing=0.06)
    fig.add_trace(go.Scatter(y=mid, name='mid (gridded)', line=dict(color='gray')), row=1, col=1)
    fig.add_trace(go.Scatter(y=eq, name='equity', line=dict(color='steelblue')), row=2, col=1)
    fig.add_trace(go.Scatter(y=pos, name='inventory', line=dict(color='mediumpurple')), row=3, col=1)
    fig.update_layout(title='BacktestL1: a quotes-only maker (heuristic fills), inventory capped',
                      yaxis=dict(title='mid'), yaxis2=dict(title='equity ($)'),
                      yaxis3=dict(title='inventory'),
                      margin=dict(l=20, r=20, t=60, b=20),
                      legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1))
    fig.show()
```
