---
name: PortfolioReport
title: Running report columns for a multi-asset backtest
implementation_family: fin
topics:
- backtesting
- portfolio
tags:
- backtest
- portfolio
- pnl
- drawdown
- turnover
- sharpe
- sharpe ratio
- performance
- backtest statistics
short: Reduce a multi-asset backtest engine output into causal portfolio-level report columns.
inputs: 1
outputs: 6
parameters: []
nan_policy: nan-aware
see_also:
- BacktestReport
- BacktestPriceTarget
- backtest_report
output_names:
- drawdown
- cum_cost
- turnover
- trades
- max_drawdown
- sharpe
---

# `PortfolioReport`

## Description

`PortfolioReport` reads the output of a portfolio backtest engine, with shape `(time,
assets, 4)`, where the last axis is `[equity, pnl, position, cost]`. It combines the
assets and returns a `(time, 6)` array with the same columns as
[`BacktestReport`](BacktestReport.md): `[drawdown, cum_cost, turnover, trades,
max_drawdown, sharpe]`.

Equity, PnL, and cost are summed across assets. Turnover is the sum of absolute
per-asset position changes, and trades counts assets whose position changed; this
keeps offsetting commodity legs visible instead of hiding them in net position.
The asset count is fixed after `reset()`. A row containing any `NaN` is skipped: the report row is all-`NaN` and the state
holds.

Positions are in units of each asset, so summing their changes only makes sense
when the units are comparable. For a book of assets with different prices, pass
a `(time, assets)` price array as a second argument: `PortfolioReport()(out,
prices)`. Turnover is then the traded notional, the running sum of
`abs(position change * price)` over the assets, in the money units of equity and
cost. A price is read only for an asset whose position changed on that row, so
a `NaN` price for an asset that is not quoted and not traded is not used. A
`NaN` price for an asset that traded skips the row.

The same operator accepts a single `(assets, 4)` row, with `(assets,)` prices,
or an iterable of rows for incremental use. With prices, a stream is two
iterables, one of rows and one of price rows, or one iterable of `(row, prices)`
pairs. The first row after `reset()` fixes the asset count and whether rows come
with prices; a later row that differs raises an error.


<!-- NAN_FOOTNOTE_START -->
## NaN handling

**Policy: `nan-aware`.** This function is built to process `NaN` inputs, as the Description explains. See [NaN and warmup](../nan_and_warmup.md).
<!-- NAN_FOOTNOTE_END -->

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

### Turnover as traded notional

```python
import numpy as np
from screamer import PortfolioReport

out = np.zeros((3, 2, 4))                       # (time, assets, 4)
out[:, :, 2] = [[0, 0], [1, 2], [1, 0]]         # positions in units
prices = np.array([[10.0, 1.0], [10.0, 1.0], [20.0, 1.0]])
PortfolioReport()(out)[:, 2]                    # [0, 3, 5]: units
PortfolioReport()(out, prices)[:, 2]            # [0, 12, 14]: notional
```
