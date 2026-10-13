---
name: BacktestPortfolioOHLCTarget
title: Backtest a portfolio of target weights on OHLC bars
implementation_family: fin
topics:
- backtesting
- portfolio
tags:
- backtest
- portfolio
- rebalancing
- weights
- multi-asset
- exposure
- simulation
- pnl
- equity curve
- rebalance
- portfolio weights
- transaction costs
short: Backtest K assets traded toward per-asset targets from one shared capital account on
  OHLC bars, executed delay bars later at a chosen bar price.
inputs: 5
outputs: 4
parameters:
- name: fee
  type: float|list[float]
  default: 0.0
  description: Fractional fee on the traded notional, one value for all assets or one per
    asset.
- name: slippage
  type: float|list[float]
  default: 0.0
  min: 0.0
  description: Adverse price offset on every fill, read in `slippage_unit`, one value for
    all assets or one per asset.
- name: slippage_unit
  type: str
  default: fraction
  description: 'Unit of `slippage`: "fraction" of the reference price (5 bps = 5e-4), "ticks"
    of `tick_size`, or "price".'
- name: tick_size
  type: float|list[float]
  default: 0.0
  min: 0.0
  description: Price grid per asset (0 = continuous). Slippage offsets round up to whole ticks.
- name: multiplier
  type: float|list[float]
  default: 1.0
  min: 0.0
  description: Contract multiplier per asset, applied to pnl, notional, and proportional fees.
- name: fee_per_contract
  type: float|list[float]
  default: 0.0
  description: Fixed fee per contract traded, one value for all assets or one per asset.
- name: size
  type: str
  default: weight
  description: 'What a target means: "weight" (a fraction of portfolio equity), "notional"
    (a currency amount), or "units" (a position).'
- name: initial_capital
  type: float
  default: 1.0
  min: 0.0
  description: Starting capital. Portfolio equity is this plus the summed pnl of all assets;
    weights are fractions of it.
- name: rebalance_band
  type: float
  default: 0.0
  min: 0.0
  description: Trade a held asset back to its target when its holding drifts from it by more
    than this fraction of equity (0 = never). Needs size "weight" or "notional".
- name: rebalance
  type: str
  default: drifted
  description: 'When the band trips: "drifted" trades only the assets beyond it, "all" rebalances
    every asset that has a target.'
- name: max_gross
  type: float
  default: .inf
  description: Largest gross exposure, sum of absolute position values over equity. Orders
    that exceed it are scaled down. The default, inf, is no cap.
- name: max_net
  type: float
  default: .inf
  min: 0.0
  description: Largest absolute net exposure, sum of position values over equity. Orders that
    exceed it are scaled down. The default, inf, is no cap.
- name: fill_at
  type: str
  default: open
  description: 'Execution price on the execution bar: "open", "close", "typical" ((high +
    low + close) / 3), or "ohlc4".'
- name: delay
  type: int
  default: 1
  min: 0
  description: Bars between the bar a target is decided on (at its close) and the bar it executes
    on. 0 needs fill_at "close".
nan_policy: nan-aware
see_also:
- BacktestPortfolioTarget
- PortfolioReport
- CrossRank
- CrossNormalize
- BacktestOHLCTarget
output_names:
- equity
- pnl
- position
- cost
---

# `BacktestPortfolioOHLCTarget`

## Description

`BacktestPortfolioOHLCTarget` is the bar version of
[`BacktestPortfolioTarget`](BacktestPortfolioTarget.md). Its inputs are five
`(T, K)` arrays: `target`, `open`, `high`, `low`, and `close`. A target decided
on bar `t`, at its close, executes `delay` bars later (default 1) at that bar's
`fill_at` price (default `"open"`). `delay=0` is allowed only with
`fill_at="close"`. Positions are marked to each bar's close.

With one asset, `size="units"`, and `initial_capital=0`, it gives the same
result as [`BacktestOHLCTarget`](BacktestOHLCTarget.md).

### Targets and sizing

Each step gives one target per asset. `size` says what it means:

- `"weight"` (default): a fraction of portfolio equity. The asset's position
  becomes `weight * equity / (price * multiplier)` units.
- `"notional"`: a currency amount, `notional / (price * multiplier)` units.
- `"units"`: the position itself.

Portfolio equity is `initial_capital` plus the summed pnl of all assets, marked
at the execution prices, so weights compound over the whole book. With the
default `initial_capital=1.0` the summed equity column reads as the
cumulative return.

### Rebalancing

A finite target is an order: the asset trades to it on its execution step. A
`NaN` target keeps the asset's position. So a rebalancing schedule is a target
array with weights on the rebalance steps and `NaN` in between, and a signal
that changes every step trades every step.

Between orders, positions drift away from their target weights as prices
move. With `rebalance_band > 0`, an asset whose holding is more than
`rebalance_band * equity` away from its target value is traded back to it.
`rebalance="drifted"` trades only those assets; `"all"` rebalances every asset
that has a target as soon as one of them is beyond the band.

### Exposure limits

`max_gross` and `max_net` limit the book after each step's orders, as multiples
of equity: gross is the sum of absolute position values, net their sum. When
the orders would exceed a limit, every asset trading on that step has its
target scaled down by the same factor. Assets that do not trade keep their
positions.

### Costs and missing prices

`fee`, `slippage`, `tick_size`, `multiplier`, and `fee_per_contract` take one
value for all assets or a list with one value per asset. A buy fills at the
reference price plus the slippage and a sell at the price minus it; the `cost`
column reports the slippage and fees.

An asset whose price is `NaN` on a step (not yet listed, halted, or not quoted)
does not trade or mark on that step. Its output row repeats its equity with
zero pnl and cost, so the portfolio sum stays defined. An order that falls due on such a step
is kept and executes at the asset's next quoted bar, unless a newer target for
that asset arrives first.

### Output

The output has shape `(T, K, 4)`: `[equity, pnl, position, cost]` for each
asset, where `equity` is the asset's cumulative pnl. The portfolio's equity is
`initial_capital + out[:, :, 0].sum(axis=1)`, and
[`PortfolioReport`](PortfolioReport.md) reduces the output to portfolio-level
drawdown, turnover, and Sharpe.

A `(T, K)` input is `T` steps. One row of `K` values per input is one step,
which is how the engine runs live: call it once per step with that step's
targets and prices. The number of assets is fixed by the first step after
`reset()`.

The inputs can also come as one packed `(T, K, 5)` array, the fields last
in the order of the separate arguments (`target`, `open`, `high`, `low` and `close`), as for every
multi-input operator: `np.stack([...], axis=-1)` gives the same results as
the separate arrays, bit for bit. In JavaScript the packed form is an
`NdArray` of shape `[T, K, 5]`.


<!-- NAN_FOOTNOTE_START -->
## NaN handling

**Policy: `nan-aware`.** This function is built to process `NaN` inputs, as the Description explains. See [NaN and warmup](../nan_and_warmup.md).
<!-- NAN_FOOTNOTE_END -->

## Examples

### Usage example

```{eval-rst}
.. plotly::
    :include-source: True

    import numpy as np
    import plotly.graph_objects as go
    from screamer import BacktestPortfolioOHLCTarget, CrossNormalize, CrossRank

    rng = np.random.default_rng(8)
    T, K = 756, 8
    drift = rng.normal(0, 4e-4, K)
    r = np.zeros((T, K))
    for t in range(1, T):                          # returns with some persistence
        r[t] = 0.05 * r[t - 1] + drift + rng.standard_normal(K) * 0.012
    close = 100 * np.exp(np.cumsum(r, axis=0))
    open_ = np.vstack([close[:1], close[:-1]])
    high, low = np.maximum(open_, close) * 1.003, np.minimum(open_, close) * 0.997

    momentum = np.full((T, K), np.nan)
    momentum[60:] = np.log(close[60:] / close[:-60])
    weights = CrossNormalize(gross=1.0, net=0.0)(CrossRank()(momentum))
    weights[np.arange(T) % 5 != 0] = np.nan        # rebalance weekly

    engine = BacktestPortfolioOHLCTarget(fee=5e-4, slippage=2e-4)
    out = engine(weights, open_, high, low, close)
    equity = 1.0 + out[:, :, 0].sum(axis=1)
    gross = np.abs(out[:, :, 2] * close).sum(axis=1) / equity

    fig = go.Figure()
    fig.add_trace(go.Scatter(y=equity, mode='lines', name='equity'))
    fig.add_trace(go.Scatter(y=gross, mode='lines', name='gross exposure', yaxis='y2'))
    fig.update_layout(title='Dollar-neutral momentum on eight assets, rebalanced weekly at the next open',
                      xaxis_title='Day', yaxis_title='Equity (initial capital 1)',
                      yaxis2=dict(overlaying='y', side='right', title='Gross exposure'),
                      margin=dict(l=20, r=20, t=80, b=20),
                      legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1))
    fig.show()
```

<!-- HELP_END -->
