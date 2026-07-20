---
name: BacktestL1TradesOrders
title: Backtest a market maker against L1 quotes and trades
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
- trades
- tape
- inventory
- transaction cost
- risk
short: "Backtest a two-sided market maker against top-of-book quotes with a trade tape driving the fills, into a costed equity curve."
inputs: 10
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
  default: touch
  description: Fill rule for a trade at your price. "touch" fills a participation partial; "breach" fills only on a trade through.
- name: participation_ratio
  type: float
  default: 1.0
  min: 0.0
  max: 1.0
  description: Fraction of an at-price trade's size captured (front-of-queue at 1.0).
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
- BacktestL1Orders
- BacktestTradesOrders
- BacktestOHLCTarget
- backtest_report
---

# `BacktestL1TradesOrders`

## Description

`BacktestL1TradesOrders` is the preferred market-making backtest: it takes both the
top-of-book quote and the trade tape, so fills come from actual executions rather
than inferred from quote-size changes. Each event carries the strategy's own quote
`(bid_price, bid_size, ask_price, ask_size)`, the market top-of-book
`(market_bid, market_ask, market_bid_size, market_ask_size)`, and a print
`(trade_price, trade_size)` (align the three streams with `combine_latest` upstream).

Quotes mark the position (to the mid) and seed context; **trades drive the
fills**. A sell-print through `bid_price` fills the full remaining, a sell-print at
`bid_price` fills `min(remaining, participation_ratio * trade_size)`, both maker
fills at `bid_price` paying `maker_fee`. A market quote that crosses your resting
price with no explaining trade is the run-over fallback (a full maker fill), and a
quote submitted already crossing the spread is a taker (fills at the market, walking
`tick_size` for the overflow, paying `taker_fee`). The sell side is symmetric.
Fills are capped to `[min_position, max_position]`.

Input contract: quotes are as-of state (forward-fill them upstream); **trades are
not forward-filled**. A `NaN` trade is a quote-only update (mark, no fill), which
is `nan_policy: ignore`, so each real trade appears on exactly one row and fills at
most once. There is no double-counting and no fill-versus-cancel ambiguity.

Inputs are `(bid_price, bid_size, ask_price, ask_size, market_bid, market_ask,
market_bid_size, market_ask_size, trade_price, trade_size)`, where the first four
slots carry the strategy's own resting quotes and the next four carry the top-of-book
market quote. It emits the four positional columns shared
by the backtest family: `0 = equity` (cumulative dollar PnL), `1 = pnl` (per
event), `2 = position`, and `3 = cost` (per event). A `NaN` in the market quote
skips the event; a `NaN` own-quote price means that side is not quoted.
[`backtest_report`](backtest_report.md) summarizes the resulting equity curve.

## Limitations

Queue position is not tracked from order identity (that needs an L3 / market-by-order
feed), so at-price fills use `participation_ratio` as a front-of-queue proxy.
Orders are counterfactual (zero volume, no market impact); a marketable order fills
beyond the displayed size only under the `tick_size` slippage assumption.

## Examples

### Usage plot

```{eval-rst}
.. plotly::
    :include-source: True

    import numpy as np
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from screamer import BacktestL1TradesOrders

    rng = np.random.default_rng(1)
    n = 600
    t = np.arange(n)
    mid = 100 + 2 * np.sin(2 * np.pi * t / n * 3) + rng.standard_normal(n) * 0.2
    half = 0.25
    bid, ask = mid - half, mid + half
    five, one = np.full(n, 5.0), np.ones(n)

    # quote at the touch on both sides; trades print at the bid (a seller hits it) or
    # the ask (a buyer lifts it), and those prints drive the fills.
    at_ask = rng.standard_normal(n) > 0
    trade_price = np.where(at_ask, ask, bid)
    trade_size = np.abs(rng.standard_normal(n)) + 0.5

    out = BacktestL1TradesOrders(fill="touch", maker_fee=-0.0001, participation_ratio=0.3,
                           max_position=15.0, min_position=-15.0)(
        bid, one, ask, one, bid, ask, five, five, trade_price, trade_size)
    eq, pos = out[:, 0], out[:, 2]

    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.4, 0.3, 0.3],
                        vertical_spacing=0.06)
    fig.add_trace(go.Scatter(y=mid, name='mid', line=dict(color='gray')), row=1, col=1)
    fig.add_trace(go.Scatter(y=eq, name='equity', line=dict(color='steelblue')), row=2, col=1)
    fig.add_trace(go.Scatter(y=pos, name='inventory', line=dict(color='seagreen')), row=3, col=1)
    fig.update_layout(title='BacktestL1TradesOrders: fills driven by the trade tape, inventory bounded',
                      yaxis=dict(title='mid'), yaxis2=dict(title='equity ($)'),
                      yaxis3=dict(title='inventory'),
                      margin=dict(l=20, r=20, t=60, b=20),
                      legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1))
    fig.show()
```
