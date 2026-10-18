---
name: LiquidationDistance
title: Distance to liquidation of a leveraged position
implementation_family: fin
topics:
- backtesting
- risk
tags:
- liquidation
- leverage
- margin
- isolated margin
- maintenance margin
- perpetual
- futures
- risk
- backtest
short: How far each bar's worst price is from the liquidation price of an isolated-margin position
  at a given leverage.
inputs: 4
outputs: 3
parameters:
- name: leverage
  type: float
  default: 10.0
  required: true
  description: Leverage of the position. The margin posted at each fill is |size| * entry / leverage.
    More than 1.
- name: maintenance
  type: float
  default: 0.005
  min: 0.0
  description: Maintenance-margin rate, as a fraction of the position value. 0 or more and less than
    1 / leverage.
nan_policy: nan-aware
see_also:
- BacktestCarry
- backtest_trades
- BacktestOHLCTarget
- Drawdown
output_names:
- distance
- liq_price
- breached
---

# `LiquidationDistance`

## Description

`LiquidationDistance` measures how close an isolated-margin position comes to
liquidation, bar by bar. Its inputs are the signed `position` and the `open`,
`high`, and `low` of the price that triggers liquidation, ideally mark-price
bars. It only measures: it does not close positions or change any pnl, so use
it to choose a leverage that leaves a buffer.

Pass the position a backtest engine reports, column 2 of its
`[equity, pnl, position, cost]` output, which is already filled at that bar's
open. A raw target is decided at a bar's close and fills at the next open, so
passing it here would place each fill one bar early, a look-ahead.

Position changes follow exchange accounting, with every fill at the bar's open:

- From flat, or on a sign flip, the position is a fresh entry at the open.
- A same-sign increase blends the entry,
  `entry = (|p0| * e0 + (|p1| - |p0|) * open) / |p1|`, and posts margin on the
  added size.
- A same-sign reduce keeps the entry, and so the liquidation price, unchanged.

The position holds `|size| * entry / leverage` of margin with no top-up, which
puts its liquidation price at

- long: `entry * (1 - 1 / leverage + maintenance)`
- short: `entry * (1 + 1 / leverage - maintenance)`

The three outputs, per bar:

- `distance`: the adverse move still needed from the bar's worst price to reach
  the liquidation price, as a fraction of that price. For a long it is
  `(low - liq_price) / low`, for a short `(liq_price - high) / high`. 0 or less
  means the bar reached the liquidation price.
- `liq_price`: the liquidation price of the current position.
- `breached`: 1 from the first bar that reached the liquidation price until the
  position goes flat or flips, 0 before it. A resize does not clear it, since
  the position would have been liquidated.

All three are `NaN` while the position is flat. For example, a long of 1
entered at 100 with `leverage=10` and the default `maintenance=0.005` has
`liq_price = 100 * (1 - 0.1 + 0.005) = 90.5`; a bar with a low of 95 then has
`distance = (95 - 90.5) / 95 = 0.047`. Adding 1 more at an open of 110 blends the
entry to 105 and moves `liq_price` to `105 * 0.905 = 95.025`.

A `NaN` position holds the previous position. A `NaN` open, high, or low gives
a `NaN` row and leaves the state as it is, except that a position change on
that bar is still taken. A fill that needs the open, a fresh entry or an
increase, then happens at the next finite open.

<!-- NAN_FOOTNOTE_START -->
## NaN handling

**Policy: `nan-aware`.** This function is built to process `NaN` inputs, as the Description explains. See [NaN and warmup](../nan_and_warmup.md).
<!-- NAN_FOOTNOTE_END -->

## Examples

### Usage plot

```{eval-rst}
.. plotly::
    :include-source: True

    import numpy as np
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from screamer import BacktestOHLCTarget, LiquidationDistance

    rng = np.random.default_rng(7)
    n = 24 * 30                                      # 30 days of hourly bars
    close = 30000 * np.exp(np.cumsum(rng.standard_normal(n) * 0.004))
    open_ = np.r_[close[0], close[:-1]]
    high = np.maximum(open_, close) * (1 + np.abs(rng.standard_normal(n)) * 0.002)
    low = np.minimum(open_, close) * (1 - np.abs(rng.standard_normal(n)) * 0.002)
    target = np.where(np.arange(n) % 240 < 200, 1.0, 0.0)   # long, flat, long, ...

    # The engine fills each target at the next open; pass the position it reports.
    position = np.asarray(BacktestOHLCTarget()(target, open_, high, low, close))[:, 2]

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08)
    fig.add_trace(go.Scatter(y=close, name='close', line=dict(color='gray')), row=1, col=1)
    for leverage, color in [(5, 'steelblue'), (20, 'crimson')]:
        distance, liq_price, breached = np.asarray(
            LiquidationDistance(leverage=leverage)(position, open_, high, low)).T
        fig.add_trace(go.Scatter(y=liq_price, name=f'liquidation price, {leverage}x',
                                 line=dict(color=color, dash='dot')), row=1, col=1)
        fig.add_trace(go.Scatter(y=distance, name=f'distance, {leverage}x',
                                 line=dict(color=color)), row=2, col=1)
    fig.add_hline(y=0, line=dict(color='black', width=1), row=2, col=1)
    fig.update_yaxes(title_text='price', row=1, col=1)
    fig.update_yaxes(title_text='distance', tickformat='.0%', row=2, col=1)
    fig.update_layout(title='LiquidationDistance: the same long at 5x and at 20x',
                      xaxis2_title='hour', margin=dict(l=20, r=20, t=60, b=20),
                      legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1))
    fig.show()
```

<!-- HELP_END -->
