---
name: BacktestPriceTarget
title: Backtest a target position on a value series
implementation_family: fin
topics:
- backtesting
tags:
- backtest
- pnl
- mark to market
- equity curve
- transaction cost
- strategy
- risk
short: "Backtest a target position against a value series (price/mark) into a costed mark-to-market equity curve."
inputs: 2
outputs: 4
parameters:
- name: spread
  type: float
  default: 0.0
  min: 0.0
  description: Fractional bid-ask spread crossed on each trade (e.g. 0.0005 = 5 bps). Default 0 is frictionless.
- name: fee
  type: float
  default: 0.0
  description: Fractional taker fee charged on the traded notional.
- name: min_position
  type: float
  default: -.inf
  description: Lower bound on the target position. Signals below this value are clamped to it.
- name: max_position
  type: float
  default: .inf
  description: Upper bound on the target position. Signals above this value are clamped to it.
- name: multiplier
  type: float
  default: 1.0
  min: 0.0
  description: Contract multiplier applied to mark-to-market PnL, fill slippage, and proportional fees.
- name: fee_per_contract
  type: float
  default: 0.0
  description: Fixed fee (or rebate when negative) charged per contract on each fill.
- name: slippage
  type: float
  default: 0.0
  min: 0.0
  description: Adverse price offset on every taker (market) fill, read in `slippage_unit`. A buy fills higher and a sell lower; resting limit fills never slip.
- name: slippage_unit
  type: str
  default: fraction
  description: 'Unit of `slippage`: "fraction" of the reference price (5 bps = 5e-4), "ticks" of `tick_size`, or "price" (an absolute amount).'
- name: tick_size
  type: float
  default: 0.0
  min: 0.0
  description: Instrument price grid (0 = continuous). Slippage offsets round up to whole ticks; the `"ticks"` slippage unit counts in this step.
- name: stop_loss
  type: float
  default: 0.0
  min: 0.0
  description: Exit when the price moves this fraction against the average entry fill (0.05 = 5%). 0 turns it off.
- name: take_profit
  type: float
  default: 0.0
  min: 0.0
  description: Exit when the price moves this fraction in favour of the average entry fill. 0 turns it off.
- name: trailing_stop
  type: float
  default: 0.0
  min: 0.0
  description: Exit when the price falls this fraction from the best price since entry (the highest for a long, the lowest for a short). 0 turns it off.
- name: size
  type: str
  default: units
  description: 'How to read the target: "units" (a position), "notional" (a currency amount), or "weight" (a fraction of initial_capital + equity). It is converted to units at the execution price.'
- name: initial_capital
  type: float
  default: 0.0
  min: 0.0
  description: Starting capital for size="weight" and for rebalance_band; equity is initial_capital plus the backtest's cumulative pnl.
- name: rebalance_band
  type: float
  default: 0.0
  min: 0.0
  description: With size="notional" or "weight", recompute the units when the holding drifts from the target by more than this fraction of equity (0.05 = 5 points of weight). 0 keeps the units until the target value changes.
nan_policy: ignore
see_also:
- Drawdown
- MaxDrawdown
- RollingInfoRatio
---

# `BacktestPriceTarget`

`multiplier` converts a price move into currency per contract. `fee_per_contract`
is added to the existing proportional taker fee on every fill and may be negative
for a rebate; defaults preserve the previous behavior.

## Description

`BacktestPriceTarget` turns a target position on a value series (a price or mark)
into a costed profit-and-loss curve by reaching that target through taker liquidity.
The `signal` is the target position in units (its sign is long, short, or flat, any
magnitude); `price` is the mark (mid). Each bar the position moves to the signal via
a market order that crosses half of the fractional `spread` (a buy fills at
`price * (1 + spread/2)`, a sell at `price * (1 - spread/2)`) and pays the
fractional `fee` on the traded notional. `slippage` moves the fill further against
the trader, in the unit set by `slippage_unit` (`"fraction"`, `"ticks"` of
`tick_size`, or `"price"`). With the defaults `spread = fee = slippage = 0` the
backtest is frictionless.

The fill happens at the same step's `price`. If the signal is computed from that
same price (for example from a close), this assumes you can trade at the price
you just observed. To trade one step later, lag the signal, or use
`BacktestOHLCTarget`, which defers execution to the next bar.

A target outside `[min_position, max_position]` is clamped to the nearest boundary
before the order is computed; the default bounds are unbounded so the behavior is
unchanged when the cap is not set.

It emits four positional columns: `0 = equity` (cumulative dollar PnL), `1 = pnl`
(per bar), `2 = position`, and `3 = cost` (per bar). It is causal: the signal at
`t` enters PnL only at `t+1`, through the position it sets, so a future signal
never changes a past row. A `NaN` signal or price skips the bar (an all-`NaN` row,
the position held across the gap; `nan_policy: ignore`).

The equity curve feeds the existing risk operators directly (dollar drawdown is
`equity - CumMax(equity)`), and [`backtest_report`](backtest_report.md) bundles
the common statistics (drawdown, cost, turnover, trades, Sharpe) in one call.

**Sizing.** With `size="notional"` the target is a currency amount and with
`size="weight"` a fraction of equity (`initial_capital` plus the backtest's
cumulative pnl, marked at the execution price), so returns compound. The engine
converts the target to units at the execution reference price when the target
value changes, and keeps those units while it stays the same, so a constant
weight does not trade on every price move. Set `rebalance_band` to trade back to
the target when the holding drifts by more than that fraction of equity.

**Risk exits.** `stop_loss`, `take_profit`, and `trailing_stop` close the position
on the step whose price crosses a level measured from the average entry fill (or,
for the trailing stop, from the best price since entry). The exit fills at that
step's price, as a market order. After an exit, targets on the same side are
ignored until the target turns flat or reverses.

## Examples

### Usage plot

```{eval-rst}
.. plotly::
    :include-source: True

    import numpy as np
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from screamer import BacktestPriceTarget, RollingMean

    rng = np.random.default_rng(0)
    n = 500
    price = 100 + np.cumsum(rng.standard_normal(n) * 0.3)
    # a trend signal: long when the fast average is above the slow one, else short
    fast = RollingMean(10)(price)
    slow = RollingMean(50)(price)
    signal = np.sign(np.nan_to_num(fast - slow))

    free = BacktestPriceTarget()(signal, price)[:, 0]                    # frictionless equity
    costed = BacktestPriceTarget(spread=0.001, fee=0.0002)(signal, price)
    eq = costed[:, 0]
    dd = eq - np.maximum.accumulate(eq)                            # running dollar drawdown

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.6, 0.4],
                        vertical_spacing=0.08)
    fig.add_trace(go.Scatter(y=free, name='equity (frictionless)',
                             line=dict(color='gray', dash='dot')), row=1, col=1)
    fig.add_trace(go.Scatter(y=eq, name='equity (with cost)',
                             line=dict(color='steelblue')), row=1, col=1)
    fig.add_trace(go.Scatter(y=dd, name='drawdown', line=dict(color='crimson'),
                             fill='tozeroy'), row=2, col=1)
    fig.update_layout(title='BacktestPriceTarget: a trend signal turned into a costed equity curve',
                      yaxis=dict(title='equity ($)'), yaxis2=dict(title='drawdown ($)'),
                      margin=dict(l=20, r=20, t=60, b=20),
                      legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1))
    fig.show()
```
