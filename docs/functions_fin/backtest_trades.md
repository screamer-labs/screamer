---
name: backtest_trades
title: backtest_trades
kind: function
short: One row per round trip of a backtest engine's output, with its entry and exit, side,
  length, pnl, and cost.
tags:
- trades
- trade list
- round trip
- holding period
- trade analysis
see_also:
- backtest_report
- BacktestStats
topics:
- backtesting
---

# `backtest_trades`

The list of round trips in a backtest engine's `[equity, pnl, position, cost]`
output, one row per trade. It returns a dict of equal-length numpy arrays:

- `entry`: the row of the trade's first nonzero position;
- `exit`: the row where the position returns to zero or flips sign, or the
  last row for a trade still open;
- `side`: `+1` long, `-1` short;
- `bars`: `exit - entry`, in rows;
- `pnl`: the pnl over the trade's rows, which is after cost;
- `cost`: the cost over the trade's rows;
- `closed`: `False` for a trade still open on the last row.

Trades are the round trips [`BacktestStats`](BacktestStats.md) counts: a
trade opens when the position leaves zero and closes when it returns to zero,
a sign flip closes one trade and opens the next, and resizing without
crossing zero stays one trade. The pnl of a flip row goes to the closing
trade. On the same output, the closed trades give `BacktestStats`'s count,
win rate, average win and loss, and average holding (rows with a `NaN` field
are skipped by both, and count in `bars` but not in `avg_holding`). Group the
rows by their `entry` to see results per month or per year; trade statistics
are not part of [`backtest_report`](backtest_report.md)'s `by=`, because a
trade can span groups.

A portfolio output `(T, K, 4)` raises; call `backtest_trades(values[:, k])`
for each asset. In JavaScript, `backtestTrades(values, { index })` returns the
same keys as typed arrays.

<!-- HELP_END -->

## Signature

`backtest_trades(values, index=None)`

`values` is the `(T, 4)` array a backtest engine emits. With `index`, `entry`
and `exit` hold its values (dates, for example) instead of row numbers.

## Examples

```{eval-rst}
.. exec_code::

    import numpy as np
    from screamer import BacktestPriceTarget, backtest_trades

    rng = np.random.default_rng(0)
    price = 100 + np.cumsum(rng.standard_normal(300))
    target = np.repeat(rng.integers(-1, 2, 15), 20).astype(float)

    trades = backtest_trades(BacktestPriceTarget(fee=1e-3)(target, price))
    for k in range(len(trades["entry"])):
        print(f"{trades['entry'][k]:4d} {trades['exit'][k]:4d} {trades['side'][k]:+d} "
              f"{trades['pnl'][k]:8.3f} {trades['closed'][k]}")
```
