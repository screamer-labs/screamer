---
name: BacktestOHLCTarget
title: Backtest a target-position strategy on OHLC bars
implementation_family: fin
topics:
- backtesting
tags:
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
short: "Backtest a target-position strategy on OHLC bars, executing market orders at the next bar's open or another bar price, with slippage (causal, no manual lag)."
inputs: 5
outputs: 4
parameters:
- name: taker_fee
  type: float
  default: 0.0
  description: Fractional fee charged on a market (taker) fill at the open.
- name: tick_size
  type: float
  default: 0.0
  min: 0.0
  description: Instrument price grid (0 = continuous). Slippage offsets round up to whole ticks; the "ticks" slippage unit counts in this step.
- name: min_position
  type: float
  default: -.inf
  description: Lower bound on the filled position. The deferred target is clamped to [min_position, max_position] before the fill executes.
- name: max_position
  type: float
  default: .inf
  description: Upper bound on the filled position. The deferred target is clamped to [min_position, max_position] before the fill executes.
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
- name: fill_at
  type: str
  default: open
  description: 'Reference price of the execution bar: "open", "close", "typical" (high + low + close) / 3, or "ohlc4" (open + high + low + close) / 4.'
- name: delay
  type: int
  default: 1
  min: 0
  description: Bars between the decision and the execution. A target from bar t trades on bar t + delay. 0 is allowed only with fill_at="close" (market-on-close of the signal bar).
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
- BacktestOHLCVTarget
- BacktestOHLCOrders
- BacktestTradesTarget
- BacktestL1Target
- backtest_report
---

# `BacktestOHLCTarget`

`multiplier` converts a price move into currency per contract. `fee_per_contract`
is added to the existing proportional taker fee on every fill and may be negative
for a rebate; defaults preserve the previous behavior.

## Description

`BacktestOHLCTarget` is a lean directional backtest on OHLC bars. It is causal by
design: the `target_position` you pass on bar t is decided from that bar's close,
and the engine executes it on bar t+1 as a market order at the open. No manual lag
is needed; feed the raw signal directly.

The deferral prevents look-ahead: a target computed from a bar's close cannot
trade within that same bar (the open already happened), so the engine holds it and
trades the next open. This mirrors `BacktestPriceTarget`, where a signal set at t
earns from t+1.

`fill_at` picks the execution price within the bar: `"open"` (default),
`"close"`, `"typical"` (high + low + close) / 3, or `"ohlc4"` (open + high + low +
close) / 4. `delay` sets how many bars later the target executes (default 1).
`delay=0` is allowed only with `fill_at="close"`: it fills at the close of the
bar the target was decided on (market-on-close).

`slippage` moves the fill against the trader, in the unit set by
`slippage_unit` (`"fraction"`, `"ticks"`, or `"price"`), and the trade pays
`taker_fee` on the traded notional. Positions mark to the close each bar. The
`cost` column holds the slippage and fees only; the move from the fill price to
the close is pnl. See [Choosing a backtest engine](choosing_a_backtest_engine.md)
for the shared slippage and cost rules.

The optional `min_position` / `max_position` parameters cap the inventory. The
deferred target is clamped to `[min_position, max_position]` before the order
size is computed, so the engine never carries a position outside that range.

Inputs are `(target_position, open, high, low, close)`. Outputs are the four
standard backtest columns: `0 = equity` (cumulative dollar PnL), `1 = pnl` (per
bar), `2 = position`, and `3 = cost` (per bar). A `NaN` in any bar field (open,
high, low, close) skips the bar (`nan_policy: ignore`). A `NaN` target keeps
the last target: the position holds once it has reached it.
[`backtest_report`](backtest_report.md) summarizes the resulting equity curve.

**Sizing.** With `size="notional"` the target is a currency amount and with
`size="weight"` a fraction of equity (`initial_capital` plus the backtest's
cumulative pnl, marked at the execution price), so returns compound. The engine
converts the target to units at the execution reference price when the target
value changes, and keeps those units while it stays the same, so a constant
weight does not trade on every price move. Set `rebalance_band` to trade back to
the target when the holding drifts by more than that fraction of equity.

**Risk exits.** `stop_loss`, `take_profit`, and `trailing_stop` close the position
when the price moves a given fraction from the average entry fill (or, for the
trailing stop, from the best price since entry). A bar that opens past a level
exits at the open; otherwise the exit is at the level the low or high reaches.
When the stop and the take-profit both fall inside one bar, the stop is taken,
because the bar does not say which came first. For the same reason, a bar's own
high (or low, for a short) raises the trailing level only from the next bar on. With `fill_at="open"` a position
opened at the open can exit later in the same bar. Exits are market orders: they
pay `taker_fee` and the `slippage` model applies. After an exit, targets on the
same side are ignored until the target turns flat or reverses, so the engine does
not re-enter on the next bar.

## Examples

### Usage plot

```{eval-rst}
.. plotly::
    :include-source: True

    import numpy as np
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from screamer import BacktestOHLCTarget, RollingMean

    rng = np.random.default_rng(5)
    n = 160
    close = 100 + np.cumsum(rng.standard_normal(n) * 0.4)
    open_ = close - rng.standard_normal(n) * 0.15
    high = np.maximum(open_, close) + np.abs(rng.standard_normal(n)) * 0.25
    low = np.minimum(open_, close) - np.abs(rng.standard_normal(n)) * 0.25

    # trend target decided from each close; the engine defers it one bar (no manual lag)
    fast, slow = RollingMean(5)(close), RollingMean(20)(close)
    signal = np.sign(np.nan_to_num(fast - slow))      # +1 long / -1 short, on each close

    out = BacktestOHLCTarget(taker_fee=0.0002)(signal, open_, high, low, close)
    eq, pos = out[:, 0], out[:, 2]

    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.5, 0.2, 0.3],
                        vertical_spacing=0.05)
    fig.add_trace(go.Candlestick(open=open_, high=high, low=low, close=close,
                                 name='bars', showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(y=signal, name='signal', line=dict(color='indigo', shape='hv')), row=2, col=1)
    fig.add_trace(go.Scatter(y=eq, name='equity', line=dict(color='steelblue')), row=3, col=1)
    fig.update_layout(title='BacktestOHLCTarget: trend signal deferred one bar and traded at the open',
                      yaxis=dict(title='price'), yaxis2=dict(title='signal'),
                      yaxis3=dict(title='equity ($)'),
                      xaxis_rangeslider_visible=False,
                      margin=dict(l=20, r=20, t=60, b=20),
                      legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1))
    fig.show()
```
