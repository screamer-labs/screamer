---
name: backtest_report
title: backtest_report
kind: function
short: Running statistics and a summary for a backtest engine's output.
topics:
- backtesting
---

# `backtest_report`

Free helper: takes a backtest engine's `[equity, pnl, position, cost]` output (the
four positional columns that [`BacktestSignal`](BacktestSignal.md) and the other
backtest engines emit) and returns `(running, summary)`.

- `running` is a pandas `DataFrame`, one row per bar, with the engine columns plus
  the running `drawdown` (dollar), `cum_cost`, `turnover` (units traded), and
  `trades` (count). Each is a causal series whose last value is the summary.
- `summary` is a pandas `Series` of the final statistics: `total_pnl`,
  `max_drawdown`, `total_cost`, `turnover`, `num_trades`, and `sharpe`.

It only aggregates the engine's outputs (no operator logic of its own), so it
works for every engine in the backtest family.

<!-- HELP_END -->

## Signature

`backtest_report(values, index=None)`

`values` is the `(T, 4)` array a backtest engine emits. `index` optionally labels
the rows (defaults to a `RangeIndex`).

## Example

```{eval-rst}
.. exec_code::

    import numpy as np
    from screamer import BacktestSignal, backtest_report

    price = 100 + np.cumsum(np.random.default_rng(0).standard_normal(500) * 0.3)
    signal = np.sign(np.random.default_rng(1).standard_normal(500))

    running, summary = backtest_report(BacktestSignal(spread=0.0005)(signal, price))
    print(summary.round(4).to_string())
    print()
    print("last running row:", running.iloc[-1].round(4).to_dict())
```
