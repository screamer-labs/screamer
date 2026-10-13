---
name: backtest_report
title: backtest_report
kind: function
short: Running statistics and a summary for a backtest engine's output.
tags:
- backtest
- performance report
- sharpe ratio
- drawdown
- turnover
- statistics
see_also:
- BacktestReport
- BacktestStats
- PortfolioReport
topics:
- backtesting
---

# `backtest_report`

Running statistics and a summary for a backtest. It takes a backtest engine's
`[equity, pnl, position, cost]` output (the four positional columns that
[`BacktestPriceTarget`](BacktestPriceTarget.md) and the other backtest engines
emit) and returns `(running, summary)`.

- `running` is a dict of numpy arrays: the four engine columns plus the
  `BacktestReport` node's `drawdown` (dollar), `cum_cost`, `turnover` (units
  traded), `trades` (count), `max_drawdown` (running worst), `sharpe` (running),
  and the [`BacktestStats`](BacktestStats.md) columns `exposure`, `round_trips`,
  `win_rate`, `profit_factor`, `avg_win`, `avg_loss`, `avg_holding`, `sortino`,
  and `calmar`. Each is a causal series whose last finite value is the summary.
- `summary` is a dict of the final statistics: `total_pnl`, `max_drawdown`,
  `total_cost`, `turnover`, `num_trades`, `sharpe`, and the nine `BacktestStats`
  values.

The report and the summary are computed in C++, which runs the `BacktestReport`
and `BacktestStats` nodes and reads the last finite value of each column. This
function only labels the result, and needs no pandas. Wrap `running` in a
`pandas.DataFrame` yourself if you want a frame. In JavaScript,
`backtestReport(values, { index, periodsPerYear })` returns
`{ running, summary }` with the same keys. There the index must hold numbers,
and `running.index` is a `Float64Array`; Python passes any index through.

<!-- HELP_END -->

## Signature

`backtest_report(values, index=None, periods_per_year=1.0)`

`values` is the `(T, 4)` array a backtest engine emits. `index` optionally adds an
`"index"` array to `running` for labeling the rows. `periods_per_year` annualizes
`sharpe`, `sortino`, and `calmar` (for example 252 for daily bars); the default 1
gives per-step ratios.

## Examples

```{eval-rst}
.. exec_code::

    import numpy as np
    from screamer import BacktestPriceTarget, backtest_report

    price = 100 + np.cumsum(np.random.default_rng(0).standard_normal(500) * 0.3)
    signal = np.sign(np.random.default_rng(1).standard_normal(500))

    running, summary = backtest_report(BacktestPriceTarget(spread=0.0005)(signal, price))
    for name, value in summary.items():
        print(f"{name:13s} {value:10.4f}")
    print("\nrunning columns:", list(running))
```
