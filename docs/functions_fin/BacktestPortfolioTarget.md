---
name: BacktestPortfolioTarget
title: Backtest a portfolio of target weights on prices
implementation_family: fin
topics:
- backtesting
tags:
- backtest
- portfolio
- rebalancing
- weights
- multi-asset
- exposure
short: "Backtest K assets traded toward per-asset target weights, notionals, or positions from one shared capital account, filled at each step's price."
inputs: 2
outputs: 4
parameters:
- name: fee
  type: float|list[float]
  default: 0.0
  description: Fractional fee on the traded notional, one value for all assets or one per asset.
- name: slippage
  type: float|list[float]
  default: 0.0
  min: 0.0
  description: Adverse price offset on every fill, read in `slippage_unit`, one value for all assets or one per asset.
- name: slippage_unit
  type: str
  default: fraction
  description: 'Unit of `slippage`: "fraction" of the reference price (5 bps = 5e-4), "ticks" of `tick_size`, or "price".'
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
  description: 'What a target means: "weight" (a fraction of portfolio equity), "notional" (a currency amount), or "units" (a position).'
- name: initial_capital
  type: float
  default: 1.0
  min: 0.0
  description: Starting capital. Portfolio equity is this plus the summed pnl of all assets; weights are fractions of it.
- name: rebalance_band
  type: float
  default: 0.0
  min: 0.0
  description: Trade a held asset back to its target when its holding drifts from it by more than this fraction of equity (0 = never). Needs size "weight" or "notional".
- name: rebalance
  type: str
  default: drifted
  description: 'When the band trips: "drifted" trades only the assets beyond it, "all" rebalances every asset that has a target.'
- name: max_gross
  type: float
  default: .inf
  description: Largest gross exposure, sum of absolute position values over equity. Orders that exceed it are scaled down.
- name: max_net
  type: float
  default: .inf
  min: 0.0
  description: Largest absolute net exposure, sum of position values over equity. Orders that exceed it are scaled down.
nan_policy: nan-aware
see_also:
- BacktestPortfolioOHLCTarget
- PortfolioReport
- CrossNormalize
- BacktestPriceTarget
---

# `BacktestPortfolioTarget`

## Description

`BacktestPortfolioTarget` backtests a portfolio of `K` assets that share one
capital account. Its inputs are two `(T, K)` arrays: `target`, one target per
asset and step, and `price`, each asset's price. Each order fills at the same
step's price, as in [`BacktestPriceTarget`](BacktestPriceTarget.md), so the
target at step `t` must be decided before that price is known; lag a signal
computed from it by one step. For bars with execution on a later bar, use
[`BacktestPortfolioOHLCTarget`](BacktestPortfolioOHLCTarget.md).

With one asset, `size="units"`, and `initial_capital=0`, it gives the same
result as `BacktestPriceTarget`.

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
zero pnl and cost, so the portfolio sum stays defined.

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

## Examples

### Usage example

```{eval-rst}
.. plotly::
    :include-source: True

    import numpy as np
    import plotly.graph_objects as go
    from screamer import BacktestPortfolioTarget

    rng = np.random.default_rng(5)
    T, K = 756, 4                                  # three years of daily prices
    vol = np.array([0.008, 0.012, 0.016, 0.02])
    prices = 100 * np.exp(np.cumsum(rng.standard_normal((T, K)) * vol + 3e-4, axis=0))

    weights = np.full((T, K), np.nan)
    weights[::21] = 0.25                           # equal weight, rebalanced monthly
    hold = np.full((T, K), np.nan)
    hold[0] = 0.25                                 # equal weight once, then left to drift

    fig = go.Figure()
    for name, target in [('rebalanced monthly', weights), ('bought and held', hold)]:
        out = BacktestPortfolioTarget(fee=5e-4)(target, prices)
        equity = 1.0 + out[:, :, 0].sum(axis=1)
        fig.add_trace(go.Scatter(y=equity, mode='lines', name=name))
    fig.update_layout(title='Four assets, equal weight: monthly rebalancing against buy and hold',
                      xaxis_title='Day', yaxis_title='Equity (initial capital 1)',
                      margin=dict(l=20, r=20, t=80, b=20),
                      legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1))
    fig.show()
```

<!-- HELP_END -->
