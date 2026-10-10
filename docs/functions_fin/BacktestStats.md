---
name: BacktestStats
title: Trade-level and downside statistics for a backtest
implementation_family: fin
topics:
- backtesting
tags:
- backtest
- win rate
- profit factor
- exposure
- sortino
- calmar
- trades
short: "Turn a backtest engine's [equity, pnl, position, cost] into running exposure, round-trip trade statistics, Sortino, and Calmar."
inputs: 4
outputs: 9
parameters:
- name: periods_per_year
  type: float
  default: 1.0
  min: 0.0
  description: Steps per year, used to annualize Sortino and Calmar (for example 252 for daily bars). The default 1 gives per-step ratios.
nan_policy: ignore
see_also:
- BacktestReport
- backtest_report
- position_test
---

# `BacktestStats`

## Description

`BacktestStats` complements [`BacktestReport`](BacktestReport.md). It reads a
backtest engine's per-step `[equity, pnl, position, cost]` and returns nine
running columns, in order:

0. `exposure`: the share of steps with a nonzero position.
1. `round_trips`: the number of closed trades.
2. `win_rate`: the share of closed trades with positive pnl.
3. `profit_factor`: the gross pnl of winning trades divided by the gross loss of
   losing trades.
4. `avg_win`: the mean pnl of a winning trade.
5. `avg_loss`: the mean pnl of a losing trade (`<= 0`).
6. `avg_holding`: the mean number of steps a closed trade held a position.
7. `sortino`: `mean(pnl) / downside deviation`, times `sqrt(periods_per_year)`.
   The downside deviation is `sqrt(mean(min(pnl, 0)^2))` over all steps.
8. `calmar`: `mean(pnl) * periods_per_year / |max drawdown|`, with the dollar
   drawdown of the equity curve.

A trade opens when the position leaves zero and closes when it returns to zero.
A sign flip closes one trade and opens the next on the same step, and resizing
without crossing zero stays one trade. A trade's pnl is the sum of the step pnl
from its opening step to its closing step. The engine reports one pnl per step,
so on a flip step the whole step's pnl goes to the trade that closes. A trade
that is still open does not count yet.

Every column is a causal running value, so the full-sample statistic is the last
finite value; [`backtest_report`](backtest_report.md) puts them in its summary. A
ratio whose denominator is empty (no losing trade, no drawdown) is `NaN`.
`nan_policy: ignore`: a row with any `NaN` field is skipped, the state holds, and
the output row is `NaN`.

## Examples

### Usage plot

```{eval-rst}
.. plotly::
    :include-source: True

    import numpy as np
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from screamer import BacktestOHLCTarget, BacktestStats, EwMean

    rng = np.random.default_rng(3)
    n = 1500
    r = np.zeros(n)
    e = rng.standard_normal(n) * 0.01
    for t in range(1, n):
        r[t] = 0.1 * r[t - 1] + e[t]
    close = 100 * np.exp(np.cumsum(r))
    open_ = np.r_[close[0], close[:-1]]
    high, low = np.maximum(open_, close) * 1.001, np.minimum(open_, close) * 0.999

    signal = np.sign(np.nan_to_num(EwMean(span=10)(r)))
    out = BacktestOHLCTarget(taker_fee=2e-4)(signal, open_, high, low, close)
    stats = BacktestStats(periods_per_year=252)(*out.T)

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08)
    fig.add_trace(go.Scatter(y=out[:, 0], name='equity', line=dict(color='steelblue')),
                  row=1, col=1)
    fig.add_trace(go.Scatter(y=stats[:, 2], name='win rate', line=dict(color='seagreen')),
                  row=2, col=1)
    fig.add_trace(go.Scatter(y=stats[:, 0], name='exposure', line=dict(color='darkorange')),
                  row=2, col=1)
    fig.update_layout(title='BacktestStats: running win rate and exposure of a trend strategy',
                      yaxis=dict(title='equity ($)'), yaxis2=dict(title='share', range=[0, 1]),
                      margin=dict(l=20, r=20, t=60, b=20),
                      legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1))
    fig.show()
```
