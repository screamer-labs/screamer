---
name: BacktestOHLC
title: Backtest a directional strategy on OHLC bars
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
- limit order
- market order
- transaction cost
- strategy
- risk
short: "Backtest a directional target-position strategy on OHLC bars, with market and limit orders, into a costed equity curve."
inputs: 6
outputs: 4
parameters:
- name: spread
  type: float
  default: 0.0
  min: 0.0
  description: Fractional bid-ask spread crossed by a market order at the open (e.g. 0.0005 = 5 bps).
- name: taker_fee
  type: float
  default: 0.0
  description: Fractional fee charged on a market (taker) fill.
- name: maker_fee
  type: float
  default: 0.0
  description: Fractional fee on a limit (maker) fill; negative for a rebate.
- name: fill
  type: str
  default: touch
  description: Limit fill rule. "touch" fills when the bar range reaches the limit, "breach" only when it trades through.
nan_policy: ignore
see_also:
- BacktestSignal
- BacktestTrades
- BacktestL1
- backtest_report
---

# `BacktestOHLC`

## Description

`BacktestOHLC` is a lean directional backtest on OHLC bars. Each bar the strategy
names a `target_position` and submits one order, both decided from data up to the
previous bar (feed a lagged target to avoid lookahead); the order is live over the
current bar. A market order (`limit_price` is `NaN`) fills at the bar's `open`,
crossing half the fractional `spread` and paying `taker_fee`. A limit order fills
only if the bar range reaches its price (`fill = "touch"` when the range touches
the level, `"breach"` when it trades through), at the limit price, paying
`maker_fee` (negative for a rebate); an unreached limit holds the position.
Positions mark to the `close`.

Inputs are `(target_position, limit_price, open, high, low, close)`. It emits the
four positional columns shared by the backtest family: `0 = equity` (cumulative
dollar PnL), `1 = pnl` (per bar), `2 = position`, and `3 = cost` (per bar). A bar
has no intra-bar path, so two-sided market-making is out of scope here (use
[`BacktestL1`](BacktestL1.md)). A `NaN` in any bar field skips the bar
(`nan_policy: ignore`); a `NaN` `limit_price` is a market order, not a skip.
[`backtest_report`](backtest_report.md) summarizes the resulting equity curve.

## Examples

### Usage plot

```{eval-rst}
.. plotly::
    :include-source: True

    import numpy as np
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from screamer import BacktestOHLC, RollingMean, Lag

    rng = np.random.default_rng(1)
    n = 400
    close = 100 + np.cumsum(rng.standard_normal(n) * 0.3)
    open_ = close - rng.standard_normal(n) * 0.1
    high = np.maximum(open_, close) + np.abs(rng.standard_normal(n)) * 0.2
    low = np.minimum(open_, close) - np.abs(rng.standard_normal(n)) * 0.2

    # trend target, lagged one bar so the decision uses only past data
    fast, slow = RollingMean(10)(close), RollingMean(50)(close)
    target = Lag(1)(np.sign(np.nan_to_num(fast - slow)))
    market = np.full(n, np.nan)                       # all-market: fill at each open

    out = BacktestOHLC(spread=0.001, taker_fee=0.0002)(target, market, open_, high, low, close)
    eq, pos = out[:, 0], out[:, 2]

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.65, 0.35],
                        vertical_spacing=0.08)
    fig.add_trace(go.Scatter(y=eq, name='equity', line=dict(color='steelblue')), row=1, col=1)
    fig.add_trace(go.Scatter(y=pos, name='position', line=dict(color='seagreen', shape='hv')), row=2, col=1)
    fig.update_layout(title='BacktestOHLC: a lagged trend target traded market-on-open',
                      yaxis=dict(title='equity ($)'), yaxis2=dict(title='position'),
                      margin=dict(l=20, r=20, t=60, b=20),
                      legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1))
    fig.show()
```
