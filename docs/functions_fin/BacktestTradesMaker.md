---
name: BacktestTradesMaker
title: Backtest a market maker against the trade tape
implementation_family: fin
topics:
- backtesting
tags:
- backtest
- pnl
- mark to market
- equity curve
- market making
- trades
- tape
- inventory
- transaction cost
- risk
short: "Backtest a two-sided market maker against the trade tape, filling resting quotes when prints cross them, into a costed equity curve."
inputs: 6
outputs: 4
parameters:
- name: maker_fee
  type: float
  default: 0.0
  description: Fractional fee on a maker fill; negative for a rebate.
- name: taker_fee
  type: float
  default: 0.0
  description: Fractional fee on a taker fill (a market order that sweeps any print).
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
  description: Accepted for interface uniformity; currently unused on the tape (see Limitations).
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
- BacktestTrades
- BacktestL1Trades
- BacktestOHLCOrders
- backtest_report
---

# `BacktestTradesMaker`

## Description

`BacktestTradesMaker` backtests a two-sided market-making strategy against the
raw trade tape. Each event is a print `(trade_price, trade_size)` paired with
the strategy's resting quote `(bid_price, bid_size, ask_price, ask_size)`.

Fill logic per event:

- **Buy side.** A resting bid at `bid_price` fills when a sell-print crosses it.
  With `fill="touch"` a print at exactly `bid_price` fills
  `min(bid_size, participation_ratio * trade_size, room)` at `bid_price`
  (maker fill, paying `maker_fee`). A print strictly below `bid_price`
  sweeps the full remaining `min(bid_size, room)` without a participation cap.
  With `fill="breach"` the at-price case is skipped; only a strict breach fills.
- **Sell side.** Symmetric for a resting ask against buy-prints.
- **Market orders.** A `NaN` bid price is treated as a market order (+inf limit),
  which sweeps on any print; that fill uses `taker_fee`. Pass `MARKET` (which
  equals `math.inf`) directly to the price argument for the same effect.
- **Inventory caps.** All fills are capped so the position stays in
  `[min_position, max_position]`.
- **Mark.** The position marks to the last trade price.

If both sides fill on the same print (possible when a spread collapses), each
side updates the account in sequence, so the final equity and position reflect
both fills.

Input contract: a `NaN` `trade_price` or `trade_size` is a no-trade event. The
`nan_policy` is `ignore`: the row emits all-NaN outputs and state is unchanged.
Only real prints drive fills. Inputs are
`(bid_price, bid_size, ask_price, ask_size, trade_price, trade_size)`.
Outputs are `(equity, pnl, position, cost)`, the four standard backtest columns
accepted by [`backtest_report`](backtest_report.md).

## Limitations

Queue priority is not tracked per order; at-price fills use `participation_ratio`
as a front-of-queue proxy.

`tick_size` is accepted for interface uniformity with `BacktestL1Trades` and
`BacktestOHLCOrders`, but it is inert on the tape: a marketable resting order
fills at the print price without walking the book, so there is no displayed-size
overflow to price. It is kept as a no-op parameter so strategies can share a
uniform parameter set across engines.

## Examples

### Usage plot

```{eval-rst}
.. plotly::
    :include-source: True

    import numpy as np
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from screamer import BacktestTradesMaker

    rng = np.random.default_rng(3)
    n = 600
    t = np.arange(n)
    mid = 100 + 2 * np.sin(2 * np.pi * t / n * 3) + rng.standard_normal(n) * 0.15
    half = 0.20
    bid, ask = mid - half, mid + half
    one = np.ones(n)

    # alternate between seller-initiated and buyer-initiated prints
    at_ask = rng.standard_normal(n) > 0
    trade_price = np.where(at_ask, ask, bid)
    trade_size = np.abs(rng.standard_normal(n)) + 0.5

    out = BacktestTradesMaker(maker_fee=-0.0001, participation_ratio=0.4,
                              max_position=12.0, min_position=-12.0)(
        bid, one, ask, one, trade_price, trade_size)
    eq, pos = out[:, 0], out[:, 2]

    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.4, 0.3, 0.3],
                        vertical_spacing=0.06)
    fig.add_trace(go.Scatter(y=mid, name='mid', line=dict(color='gray')), row=1, col=1)
    fig.add_trace(go.Scatter(y=eq, name='equity', line=dict(color='steelblue')), row=2, col=1)
    fig.add_trace(go.Scatter(y=pos, name='inventory', line=dict(color='seagreen')), row=3, col=1)
    fig.update_layout(title='BacktestTradesMaker: fills driven by the trade tape, inventory bounded',
                      yaxis=dict(title='mid'), yaxis2=dict(title='equity ($)'),
                      yaxis3=dict(title='inventory'),
                      margin=dict(l=20, r=20, t=60, b=20),
                      legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1))
    fig.show()
```
