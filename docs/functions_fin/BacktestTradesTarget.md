---
name: BacktestTradesTarget
title: Backtest a target-position strategy against the trade tape
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
- market order
- transaction cost
- strategy
- risk
short: "Backtest a target-position strategy against the trade tape, taking each print as a market fill to reach the target immediately."
inputs: 3
outputs: 4
parameters:
- name: taker_fee
  type: float
  default: 0.0
  description: Fractional fee charged on each taker fill at the print price.
- name: tick_size
  type: float
  default: 0.0
  min: 0.0
  description: Accepted for interface uniformity; currently unused on the tape.
- name: min_position
  type: float
  default: -.inf
  description: Inventory floor. The target is clamped to [min_position, max_position] before the fill size is computed.
- name: max_position
  type: float
  default: .inf
  description: Inventory ceiling. The target is clamped to [min_position, max_position] before the fill size is computed.
nan_policy: ignore
see_also:
- BacktestTradesOrders
- BacktestOHLCTarget
- BacktestL1Trades
- backtest_report
---

# `BacktestTradesTarget`

## Description

`BacktestTradesTarget` is a lean directional backtest on the raw trade tape. It is
immediate: the `target_position` you pass on event t executes on the same print,
unlike `BacktestOHLCTarget` which defers to the next bar's open.

Each print the engine computes the desired trade as
`clamp(target, min_position, max_position) - current_position` and takes it
against the current print as a marketable (taker) order. The fill pays `taker_fee`
on the traded notional and the position marks to the print price. If the target
equals the current position, no order is placed and the event is a mark-only step.

Inputs are `(target_position, trade_price, trade_size)`. Outputs are the four
standard backtest columns: `0 = equity` (cumulative dollar PnL), `1 = pnl` (per
event), `2 = position`, and `3 = cost` (per event). A `NaN` `trade_price` or
`trade_size` skips the event and returns all-NaN (`nan_policy: ignore`).
[`backtest_report`](backtest_report.md) summarizes the resulting equity curve.

The optional `min_position` and `max_position` parameters cap the inventory. The
target is clamped before the order size is computed, so the engine never carries a
position outside that range.

## Limitations

The fill model is optimistic: the full order size executes at the print price with
no market impact or slippage beyond `taker_fee`. Orders are counterfactual (zero
market impact), so the tape is unaffected by the strategy's activity.

## Examples

### Usage plot

```{eval-rst}
.. plotly::
    :include-source: True

    import numpy as np
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from screamer import BacktestTradesTarget

    rng = np.random.default_rng(5)
    n = 400
    t = np.arange(n)
    price = 100 + 2 * np.sin(2 * np.pi * t / n * 4) + np.cumsum(rng.standard_normal(n) * 0.1)
    size = np.ones(n)

    # a simple mean-reversion signal: go long when cheap, short when rich
    signal = np.sign(100 - price)

    out = BacktestTradesTarget(taker_fee=0.0002, max_position=3.0, min_position=-3.0)(
        signal, price, size)
    eq, pos = out[:, 0], out[:, 2]

    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.4, 0.3, 0.3],
                        vertical_spacing=0.06)
    fig.add_trace(go.Scatter(y=price, name='price', line=dict(color='gray')), row=1, col=1)
    fig.add_trace(go.Scatter(y=eq, name='equity', line=dict(color='steelblue')), row=2, col=1)
    fig.add_trace(go.Scatter(y=pos, name='position', line=dict(color='darkorange', shape='hv')),
                  row=3, col=1)
    fig.update_layout(title='BacktestTradesTarget: immediate taker fills on each print',
                      yaxis=dict(title='price'), yaxis2=dict(title='equity ($)'),
                      yaxis3=dict(title='position'),
                      margin=dict(l=20, r=20, t=60, b=20),
                      legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1))
    fig.show()
```
