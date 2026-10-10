---
name: backtest_report
title: backtest_report
kind: function
short: Running statistics and a summary for a backtest engine's output.
topics:
- backtesting
---

# `backtest_report`

Labels the output of the [`BacktestReport`](BacktestReport.md) node. It takes a
backtest engine's `[equity, pnl, position, cost]` output (the four positional
columns that [`BacktestPriceTarget`](BacktestPriceTarget.md) and the other backtest engines
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

The aggregation runs in the C++ `BacktestReport` node, so a pure-C++ user gets the
same statistics by calling that node directly. This wrapper only labels its
columns and reads the last row, and needs no pandas. Wrap `running` in a
`pandas.DataFrame` yourself if you want a frame.

<!-- HELP_END -->

## Signature

`backtest_report(values, index=None, periods_per_year=1.0)`

`values` is the `(T, 4)` array a backtest engine emits. `index` optionally adds an
`"index"` array to `running` for labeling the rows. `periods_per_year` annualizes
`sharpe`, `sortino`, and `calmar` (for example 252 for daily bars); the default 1
gives per-step ratios.

## Example

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
