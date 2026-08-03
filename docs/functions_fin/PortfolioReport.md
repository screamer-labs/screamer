---
name: PortfolioReport
title: Running report columns for a multi-asset backtest
implementation_family: fin
topics:
- backtesting
tags:
- backtest
- portfolio
- pnl
- drawdown
- turnover
- sharpe
short: "Reduce a multi-asset backtest engine output into causal portfolio-level report columns."
inputs: 1
outputs: 6
parameters: []
nan_policy: ignore
see_also:
- BacktestReport
- BacktestPriceTarget
- backtest_report
---

# `PortfolioReport`

## Description

`PortfolioReport` consumes the C++ backtest engine output with shape `(time,
assets, 4)`, where the last axis is `[equity, pnl, position, cost]`. It reduces
assets in C++ and returns a `(time, 6)` array with the same columns as
[`BacktestReport`](BacktestReport.md): `[drawdown, cum_cost, turnover, trades,
max_drawdown, sharpe]`.

Equity, PnL, and cost are summed across assets. Turnover is the sum of absolute
per-asset position changes, and trades counts assets whose position changed; this
keeps offsetting commodity legs visible instead of hiding them in net position.
The asset count is fixed after `reset()`. A row containing any `NaN` is skipped
(`nan_policy: ignore`) and returns an all-`NaN` report row while holding state.

The same operator accepts a single `(assets, 4)` row or an iterable of such rows
for incremental use. The batch and row paths share the same causal state machine.

## Examples

### Reduce engine output

```python
import numpy as np
from screamer import BacktestPriceTarget, PortfolioReport

price = np.array([100.0, 101.0, 99.0])
signal = np.array([1.0, 1.0, 0.0])
engine = BacktestPriceTarget()(signal, price)
engine_output = np.stack([engine, engine], axis=1)  # (time, assets, 4)
report = PortfolioReport()(engine_output)             # (time, 6)
```
