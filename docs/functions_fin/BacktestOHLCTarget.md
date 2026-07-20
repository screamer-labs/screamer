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
short: "Backtest a target-position strategy on OHLC bars, executing market orders at the next bar's open (causal, no manual lag)."
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
  description: Price increment added (subtracted) to the open for a market buy (sell) to model taker slippage.
- name: min_position
  type: float
  default: -.inf
  description: Lower bound on the filled position. The deferred target is clamped to [min_position, max_position] before the fill executes.
- name: max_position
  type: float
  default: .inf
  description: Upper bound on the filled position. The deferred target is clamped to [min_position, max_position] before the fill executes.
nan_policy: ignore
see_also:
- BacktestOHLCOrders
- BacktestTrades
- BacktestL1
- backtest_report
---

# `BacktestOHLCTarget`

## Description

`BacktestOHLCTarget` is a lean directional backtest on OHLC bars. It is causal by
design: the `target_position` you pass on bar t is decided from that bar's close,
and the engine executes it on bar t+1 as a market order at the open. No manual lag
is needed; feed the raw signal directly.

The deferral prevents look-ahead: a target computed from a bar's close cannot
trade within that same bar (the open already happened), so the engine holds it and
trades the next open. This mirrors `BacktestPriceTarget`, where a signal set at t
earns from t+1.

The market order fills at `open + tick_size` (buy) or `open - tick_size` (sell),
paying `taker_fee` on the traded notional. Positions mark to the close each bar.

The optional `min_position` / `max_position` parameters cap the inventory. The
deferred target is clamped to `[min_position, max_position]` before the order
size is computed, so the engine never carries a position outside that range.

Inputs are `(target_position, open, high, low, close)`. Outputs are the four
standard backtest columns: `0 = equity` (cumulative dollar PnL), `1 = pnl` (per
bar), `2 = position`, and `3 = cost` (per bar). A `NaN` in any bar field (open,
high, low, close) skips the bar (`nan_policy: ignore`). A `NaN` target holds the
position for the next bar without issuing a new order.
[`backtest_report`](backtest_report.md) summarizes the resulting equity curve.

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
