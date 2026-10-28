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
- portfolio
see_also:
- BacktestReport
- BacktestStats
- BacktestPortfolioOHLCTarget
- PortfolioReport
- backtest_trades
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

`values` can also be the `(T, K, 4)` output of a portfolio engine such as
[`BacktestPortfolioOHLCTarget`](BacktestPortfolioOHLCTarget.md), one
`[equity, pnl, position, cost]` record per asset. The report then describes
the book, the account the `K` assets share:

- The book's `equity`, `pnl`, and `cost` are summed over the assets. `running`
  holds these three columns in place of the four engine columns, because a
  book has no single position.
- `sharpe`, `drawdown`, `max_drawdown`, `sortino`, and `calmar` come from the
  summed book.
- `turnover` and `trades` add up the assets' position changes, as
  [`PortfolioReport`](PortfolioReport.md) does. `exposure` is the share of the
  assets' rows that hold a position, and the round-trip statistics pool the
  round trips of every asset.
- `summary["per_asset"]` is a list of `K` summaries; entry `k` equals the
  summary of `backtest_report(values[:, k, :])`.

An asset whose row has a `NaN` field is skipped on that row: it adds no pnl,
cost, or trade, and its equity holds at its last value (0 before its first
row). An asset that starts late joins the book when its data starts. A book
row is `NaN` only when every asset is skipped.

The report and the summary are computed in C++, which runs the `BacktestReport`
and `BacktestStats` nodes and reads the last finite value of each column. This
function only labels the result, and needs no pandas. Wrap `running` in a
`pandas.DataFrame` yourself if you want a frame. In JavaScript,
`backtestReport(values, { index, periodsPerYear })` returns
`{ running, summary }` with the same keys, and for a `[T, K, 4]` input the
per-asset summaries as `perAsset`. There the index must hold numbers,
and `running.index` is a `Float64Array`; Python passes any index through.

<!-- HELP_END -->

## Signature

`backtest_report(values, index=None, periods_per_year=1.0, by=None)`

`values` is the `(T, 4)` array a backtest engine emits, or the `(T, K, 4)`
array a portfolio engine emits. `index` optionally adds an
`"index"` array to `running` for labeling the rows. `periods_per_year` annualizes
`sharpe`, `sortino`, and `calmar` (for example 252 for daily bars); the default 1
gives per-step ratios.

`by` is one label per row, of any NumPy dtype, as in pandas `groupby(by=)`:
`datetime64[M]` months, integer years, or strings. With it, `summary` gains
`by_group`, a dict from label (in order of first appearance) to that group's
`total_pnl`, `total_cost`, `turnover`, `sharpe` (annualized with
`periods_per_year`), `max_drawdown` (of the cumulative pnl over the group's
rows), and `steps`. Rows with a `NaN` field are skipped. For a `(T, K, 4)`
input the groups hold the book's rows. Trade statistics are
not grouped, because a trade can span two groups; group the rows of
[`backtest_trades`](backtest_trades.md) by their `entry` instead. In
JavaScript, `backtestReport(values, { by })` returns the groups as `byGroup`,
a `Map` from label to the same six statistics.

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

```{eval-rst}
.. exec_code::

    import numpy as np
    from screamer import BacktestPriceTarget, backtest_report

    rng = np.random.default_rng(0)
    days = np.datetime64("2024-01-01") + np.arange(366)
    price = 100 + np.cumsum(rng.standard_normal(366))
    signal = np.sign(rng.standard_normal(366))

    _, summary = backtest_report(BacktestPriceTarget(fee=1e-3)(signal, price),
                                 periods_per_year=252, by=days.astype("datetime64[M]"))
    for month, s in summary["by_group"].items():
        print(month, f"{s['total_pnl']:8.2f} {s['sharpe']:6.2f} {s['steps']:4d}")
```

The book of a two-asset portfolio, and the summary of each asset:

```{eval-rst}
.. exec_code::

    import numpy as np
    from screamer import BacktestPortfolioOHLCTarget, backtest_report

    rng = np.random.default_rng(0)
    close = 100 * np.exp(np.cumsum(0.01 * rng.standard_normal((365, 2)), axis=0))
    target = np.where(rng.standard_normal((365, 2)) > 0, 0.5, -0.5)

    out = BacktestPortfolioOHLCTarget(fee=1e-3)(target, close, close, close, close)
    print(out.shape)
    running, summary = backtest_report(out, periods_per_year=365)
    print(f"book     pnl {summary['total_pnl']:8.4f}  sharpe {summary['sharpe']:6.2f}")
    for k, s in enumerate(summary["per_asset"]):
        print(f"asset {k}  pnl {s['total_pnl']:8.4f}  sharpe {s['sharpe']:6.2f}")
    print("running columns:", list(running)[:5], "...")
```
