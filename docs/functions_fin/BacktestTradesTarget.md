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
- simulation
- transaction costs
- fees
short: Backtest a target-position strategy against the trade tape, taking each print as a
  market fill to reach the target immediately.
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
  description: Instrument price grid (0 = continuous). Slippage offsets round up to whole
    ticks; the "ticks" slippage unit counts in this step.
- name: min_position
  type: float
  default: -.inf
  description: Lower bound on the position, in units. It applies after `size` has converted
    the target to units, so a weight target of -2 with min_position=-0.5 holds -0.5 units.
- name: max_position
  type: float
  default: .inf
  description: Upper bound on the position, in units. It applies after `size` has converted
    the target to units, so a weight target of 2 with max_position=0.5 holds 0.5 units.
- name: multiplier
  type: float
  default: 1.0
  min: 0.0
  description: Contract multiplier applied to mark-to-market PnL, fill slippage, and proportional
    fees.
- name: fee_per_contract
  type: float
  default: 0.0
  description: Fixed fee (or rebate when negative) charged per contract on each fill.
- name: slippage
  type: float
  default: 0.0
  min: 0.0
  description: Adverse price offset on every taker (market) fill, read in `slippage_unit`.
    A buy fills higher and a sell lower; resting limit fills never slip.
- name: slippage_unit
  type: str
  default: fraction
  description: 'Unit of `slippage`: "fraction" of the reference price (5 bps = 5e-4), "ticks"
    of `tick_size`, or "price" (an absolute amount).'
- name: stop_loss
  type: float
  default: 0.0
  min: 0.0
  description: Exit when the price moves this fraction against the average entry fill (0.05
    = 5%). 0 turns it off.
- name: take_profit
  type: float
  default: 0.0
  min: 0.0
  description: Exit when the price moves this fraction in favor of the average entry fill.
    0 turns it off.
- name: trailing_stop
  type: float
  default: 0.0
  min: 0.0
  description: Exit when the price falls this fraction from the best price since entry (the
    highest for a long, the lowest for a short). 0 turns it off.
- name: size
  type: str
  default: units
  description: 'How to read the target: "units" (a position), "notional" (a currency amount),
    or "weight" (a fraction of initial_capital + equity). It is converted to units at the
    execution price.'
- name: initial_capital
  type: float
  default: 0.0
  min: 0.0
  description: Starting capital for size="weight" and for rebalance_band; equity is initial_capital
    plus the backtest's cumulative pnl.
- name: rebalance_band
  type: float
  default: 0.0
  min: 0.0
  description: With size="notional" or "weight", recompute the units when the holding drifts
    from the target by more than this fraction of equity (0.05 = 5 points of weight). 0 keeps
    the units until the target value changes.
- name: max_participation
  type: float
  default: .inf
  min: 0.0
  description: Cap each print's fill at this share of the print's size; the rest fills on
    later prints. The default fills the whole order on any print.
nan_policy: nan-aware
see_also:
- BacktestTradesOrders
- BacktestOHLCTarget
- BacktestL1TradesOrders
- backtest_report
output_names:
- equity
- pnl
- position
- cost
---

# `BacktestTradesTarget`

## Description

`BacktestTradesTarget` is a directional backtest on the raw trade tape. It is
immediate: the `target_position` you pass on event t executes on the same print,
unlike `BacktestOHLCTarget` which defers to the next bar's open.

Each print the engine converts the target to units (see Sizing below), clamps it
to `[min_position, max_position]`, and takes the difference from the current
position against the print as a marketable (taker) order. The fill pays `taker_fee`
on the traded notional and the position marks to the print price. If the target
equals the current position, no order is placed and the event is a mark-only step.

Inputs are `(target_position, trade_price, trade_size)`. Outputs are the four
standard backtest columns: `0 = equity` (cumulative dollar PnL), `1 = pnl` (per
event), `2 = position`, and `3 = cost` (per event). A `NaN` `trade_price` or
`trade_size` skips the event and returns an all-NaN row.
[`backtest_report`](backtest_report.md) summarizes the resulting equity curve.

`min_position` and `max_position` bound the position in units. They apply after
the sizing conversion, so a weight target of 2 with `max_position=0.5` holds 0.5
units, and the engine never carries a position outside that range.

**Sizing.** With `size="notional"` the target is a currency amount and with
`size="weight"` a fraction of equity (`initial_capital` plus the backtest's
cumulative pnl, marked at the execution price), so returns compound. The engine
converts the target to units at the execution reference price when the target
value changes, and keeps those units while it stays the same, so a constant
weight does not trade on every price move. Set `rebalance_band` to trade back to
the target when the holding drifts by more than that fraction of equity.

**Participation.** `max_participation` limits each print's fill to that share of
the print's size, so a large order fills across several prints. The engine keeps
working toward the last target, and a `NaN` target continues the order.

**Risk exits.** `stop_loss`, `take_profit`, and `trailing_stop` close the position
on the print that crosses a level measured from the average entry fill (or, for
the trailing stop, from the best price since entry). The exit takes that print,
as a market order. After an exit, targets on the same side are ignored until the
target turns flat or reverses.

## Limitations

The fill model is optimistic: the full order size executes at the print price,
moved only by the `slippage` you set, with no market impact. Orders are counterfactual, so the tape is unaffected by the strategy's activity.


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
