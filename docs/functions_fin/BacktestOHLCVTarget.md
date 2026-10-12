---
name: BacktestOHLCVTarget
title: Backtest a target-position strategy on OHLCV bars with a volume limit
implementation_family: fin
topics:
- backtesting
tags:
- volume
- participation
- backtest
- pnl
- mark to market
- equity curve
- ohlc
- bars
- market order
- transaction cost
- strategy
- risk
- simulation
- transaction costs
- fees
short: Backtest a target-position strategy on OHLCV bars, filling each bar at most a set share
  of its volume.
inputs: 6
outputs: 4
parameters:
- name: max_participation
  type: float
  default: 1.0
  min: 0.0
  description: The most an order may trade on one bar, as a share of the bar's volume (0.1
    = 10%). The rest carries to the next bars.
- name: taker_fee
  type: float
  default: 0.0
  description: Fractional fee charged on a market (taker) fill at the open.
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
- name: fill_at
  type: str
  default: open
  description: 'Reference price of the execution bar: "open", "close", "typical" (high + low
    + close) / 3, or "ohlc4" (open + high + low + close) / 4.'
- name: delay
  type: int
  default: 1
  min: 0
  description: Bars between the decision and the execution. A target from bar t trades on
    bar t + delay. 0 is allowed only with fill_at="close" (market-on-close of the signal bar).
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
nan_policy: nan-aware
see_also:
- BacktestOHLCTarget
- BacktestOHLCOrders
- BacktestTradesTarget
- BacktestL1Target
- backtest_report
output_names:
- equity
- pnl
- position
- cost
---

# `BacktestOHLCVTarget`

## Description

`BacktestOHLCVTarget` is [`BacktestOHLCTarget`](BacktestOHLCTarget.md) with the
bar's volume as a sixth input and a limit on how much an order can trade per bar.
An order trades at most `max_participation * volume` units on a bar. The engine
keeps working toward the last target, so the rest of the order fills on the
following bars; a `NaN` target continues it, and a new target replaces it.

A backtest that fills any order in full at one price assumes the market would
have absorbed it. For orders that are large relative to the traded volume, the
participation limit spreads the fill over time instead, which is how such orders
are executed in practice. A common limit is 5% to 20% of volume.

Everything else is as in `BacktestOHLCTarget`: the target decided on bar t
executes `delay` bars later at the `fill_at` price, `slippage` and `taker_fee`
apply to every fill, `size` reads the target as units, a notional amount, or a
weight of equity, and `stop_loss`, `take_profit`, and `trailing_stop` add risk
exits. Risk exits are not limited by volume: an exit closes the whole position.

Inputs are `(target, open, high, low, close, volume)`. Outputs are
`[equity, pnl, position, cost]`. A `NaN` in a bar field or a `NaN` or negative
volume skips the bar.


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
    from screamer import BacktestOHLCVTarget

    rng = np.random.default_rng(3)
    n = 60
    close = 100 * np.exp(np.cumsum(rng.standard_normal(n) * 0.01))
    open_ = np.r_[close[0], close[:-1]]
    high, low = np.maximum(open_, close) * 1.002, np.minimum(open_, close) * 0.998
    volume = rng.uniform(200, 600, n)

    target = np.full(n, np.nan)
    target[5], target[35] = 1000.0, 0.0       # buy 1000 units, later sell them all

    fig = go.Figure()
    for share in (1.0, 0.25, 0.1):
        out = BacktestOHLCVTarget(max_participation=share)(target, open_, high, low, close, volume)
        fig.add_trace(go.Scatter(y=out[:, 2], name=f'{share:.0%} of volume', line=dict(shape='hv')))
    fig.update_layout(title='BacktestOHLCVTarget: a 1000-unit order filled at a share of bar volume',
                      xaxis_title='bar', yaxis_title='position (units)',
                      margin=dict(l=20, r=20, t=60, b=20),
                      legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1))
    fig.show()
```
