---
name: BacktestL1Target
title: Backtest a target-position strategy against the L1 book
implementation_family: fin
topics:
- backtesting
tags:
- backtest
- pnl
- mark to market
- equity curve
- l1
- top of book
- market order
- transaction cost
- strategy
- risk
- simulation
- transaction costs
- fees
short: Backtest a target-position strategy against the L1 book, taking each quote update as
  a market fill to reach the target immediately.
inputs: 5
outputs: 4
parameters:
- name: taker_fee
  type: float
  default: 0.0
  description: Fractional fee charged on each taker fill.
- name: tick_size
  type: float
  default: 0.0
  min: 0.0
  description: Instrument price grid (0 = continuous). A marketable order walks one tick past
    the displayed quote for the size beyond it; slippage offsets round up to whole ticks.
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
nan_policy: ignore
see_also:
- BacktestL1Orders
- BacktestL1TradesOrders
- BacktestTradesTarget
- BacktestOHLCTarget
- backtest_report
output_names:
- equity
- pnl
- position
- cost
---

# `BacktestL1Target`

## Description

`BacktestL1Target` is a directional backtest on the L1 quote stream. It is
immediate: the `target_position` you pass on event t executes on the same quote
update, unlike `BacktestOHLCTarget` which defers to the next bar's open.

Each event the engine converts the target to units (see Sizing below), clamps it
to `[min_position, max_position]`, and takes the difference from the current
position as a marketable (taker) order against the displayed L1 book. A buy order sweeps the ask
side; a sell order sweeps the bid side. The fill pays `taker_fee` on the traded
notional. If the displayed size is smaller than the desired size, the overflow is
filled at `ask + tick_size` (buy) or `bid - tick_size` (sell). Positions mark to
the mid. If the target equals the current position, no order is placed and the
event is a mark-only step.

Inputs are `(target_position, market_bid, market_ask, market_bid_size,
market_ask_size)`. Outputs are the four standard backtest columns: `0 = equity`
(cumulative dollar PnL), `1 = pnl` (per event), `2 = position`, and `3 = cost`
(per event). A `NaN` in any market field skips the event and returns an all-NaN
row. [`backtest_report`](backtest_report.md) summarizes the resulting equity
curve.

`min_position` and `max_position` bound the position in units. They apply after
the sizing conversion, so a weight target of 2 with `max_position=0.5` holds 0.5
units, and the engine never carries a position outside that range.

For a two-sided resting-order variant, see
[`BacktestL1Orders`](BacktestL1Orders.md). When a trade feed is available,
[`BacktestTradesTarget`](BacktestTradesTarget.md) uses real prints instead of
quote updates.

**Sizing.** With `size="notional"` the target is a currency amount and with
`size="weight"` a fraction of equity (`initial_capital` plus the backtest's
cumulative pnl, marked at the execution price), so returns compound. The engine
converts the target to units at the execution reference price when the target
value changes, and keeps those units while it stays the same, so a constant
weight does not trade on every price move. Set `rebalance_band` to trade back to
the target when the holding drifts by more than that fraction of equity.

**Risk exits.** `stop_loss`, `take_profit`, and `trailing_stop` close the position
on the quote update whose mid crosses a level measured from the average entry
fill (or, for the trailing stop, from the best mid since entry). The exit takes
the book, as a market order. After an exit, targets on the same side are ignored
until the target turns flat or reverses.

## Limitations

The fill model is optimistic: the order executes immediately at the displayed book
price with no queue or market impact beyond the `tick_size` overflow slippage.
Orders are counterfactual, so the book is unaffected by the strategy's activity.


<!-- NAN_FOOTNOTE_START -->
## NaN handling

**Policy: `ignore`.** A `NaN` input never corrupts the engine's state. What a `NaN` target, price or quote does on this engine is stated in the Description. See [NaN and warmup](../nan_and_warmup.md).
<!-- NAN_FOOTNOTE_END -->

## Examples

### Usage plot

```{eval-rst}
.. plotly::
    :include-source: True

    import numpy as np
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from screamer import BacktestL1Target

    rng = np.random.default_rng(6)
    n = 400
    t = np.arange(n)
    mid = 100 + 2 * np.sin(2 * np.pi * t / n * 4) + np.cumsum(rng.standard_normal(n) * 0.1)
    tick = 0.1
    market_bid, market_ask = mid - tick, mid + tick
    size = np.ones(n)

    # a simple mean-reversion signal: go long when cheap, short when rich
    signal = np.sign(100 - mid)

    out = BacktestL1Target(taker_fee=0.0002, max_position=3.0, min_position=-3.0)(
        signal, market_bid, market_ask, size, size)
    eq, pos = out[:, 0], out[:, 2]

    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.4, 0.3, 0.3],
                        vertical_spacing=0.06)
    fig.add_trace(go.Scatter(y=mid, name='mid', line=dict(color='gray')), row=1, col=1)
    fig.add_trace(go.Scatter(y=eq, name='equity', line=dict(color='steelblue')), row=2, col=1)
    fig.add_trace(go.Scatter(y=pos, name='position', line=dict(color='darkorange', shape='hv')),
                  row=3, col=1)
    fig.update_layout(title='BacktestL1Target: immediate taker fills on each L1 quote update',
                      yaxis=dict(title='mid'), yaxis2=dict(title='equity ($)'),
                      yaxis3=dict(title='position'),
                      margin=dict(l=20, r=20, t=60, b=20),
                      legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1))
    fig.show()
```
