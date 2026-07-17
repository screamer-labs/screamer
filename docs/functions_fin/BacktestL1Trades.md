---
name: BacktestL1Trades
title: Backtest a market maker against L1 quotes and trades
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
- BacktestL1
- BacktestTrades
- BacktestOHLC
- backtest_report
---

# `BacktestL1Trades`

## Description

`BacktestL1Trades` is the preferred market-making backtest: it takes both the
top-of-book quote and the trade tape, so fills come from actual executions rather
than inferred from quote-size changes. Each event carries the market quote
`(bid, ask, bid_size, ask_size)`, the strategy's own quote
`(my_bid, my_bid_size, my_ask, my_ask_size)`, and a print `(trade_price,
trade_size)` (align the three streams with `combine_latest` upstream).

Quotes mark the position (to the mid) and seed context; **trades drive the
fills**. A sell-print through `my_bid` fills the full remaining, a sell-print at
`my_bid` fills `min(remaining, participation_ratio * trade_size)`, both maker fills
at `my_bid` paying `maker_fee`. A quote that crosses your resting price with no
explaining trade is the run-over fallback (a full maker fill), and a quote
submitted already crossing the spread is a taker (fills at the market, walking
`tick_size` for the overflow, paying `taker_fee`). The sell side is symmetric.
Fills are capped to `[min_position, max_position]`.

Input contract: quotes are as-of state (forward-fill them upstream); **trades are
not forward-filled**. A `NaN` trade is a quote-only update (mark, no fill), which
is `nan_policy: ignore`, so each real trade appears on exactly one row and fills at
most once. There is no double-counting and no fill-versus-cancel ambiguity.

Inputs are `(bid, ask, bid_size, ask_size, my_bid, my_bid_size, my_ask,
my_ask_size, trade_price, trade_size)`. It emits the four positional columns shared
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
    from screamer import BacktestL1Trades, Lag

    rng = np.random.default_rng(5)
    n = 800
    mid = 100 + np.cumsum(rng.standard_normal(n) * 0.05)
    half = 0.5
    bid, ask = mid - half, mid + half
    five, one = np.full(n, 5.0), np.ones(n)

    # rest last event's touch on both sides; the trade tape (a print each event) fills them
    my_bid = np.nan_to_num(Lag(1)(bid), nan=bid[0])
    my_ask = np.nan_to_num(Lag(1)(ask), nan=ask[0])
    trade_price = mid                                    # a print at the mid each event
    trade_size = np.abs(rng.standard_normal(n)) + 0.5

    out = BacktestL1Trades(fill="touch", maker_fee=-0.0001, participation_ratio=0.3,
                           max_position=15.0, min_position=-15.0)(
        bid, ask, five, five, my_bid, one, my_ask, one, trade_price, trade_size)
    eq, pos = out[:, 0], out[:, 2]

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.6, 0.4],
                        vertical_spacing=0.08)
    fig.add_trace(go.Scatter(y=eq, name='equity', line=dict(color='steelblue')), row=1, col=1)
    fig.add_trace(go.Scatter(y=pos, name='inventory', line=dict(color='seagreen')), row=2, col=1)
    fig.update_layout(title='BacktestL1Trades: fills driven by the trade tape, inventory bounded',
                      yaxis=dict(title='equity ($)'), yaxis2=dict(title='inventory'),
                      margin=dict(l=20, r=20, t=60, b=20),
                      legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1))
    fig.show()
```
