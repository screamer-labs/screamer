---
name: walk_forward
title: walk_forward
kind: function
short: Pick the best of several strategy variants on each training window and trade it on the next test window, out of sample.
tags:
- walk forward
- walk-forward
- out-of-sample
- parameter selection
topics:
- significance
---

# `walk_forward`

Measures what a parameter search delivers out of sample. The best variant of a
strategy, chosen on the whole history, looks better than it is, because the
choice used the data it is scored on. `walk_forward` makes the choice the way
a trader would: on each training window it picks the candidate with the best
metric, and it trades that candidate on the test window that follows. The
stitched out-of-sample run is the honest estimate of the procedure "use the
variant that did best recently".

The history is split into folds. Test windows start at
`t0 = train, train + test, train + 2 * test, ...` while `t0 < T`, and each one
covers `[t0, min(t0 + test, T))`. Its training window is the `train` steps
before it, `[t0 - train, t0)`, or all of the history before it,
`[0, t0)`, with `anchored=True`.

Every candidate is backtested once over the whole history, and each training
window is scored from that run as a backtest of its own: its equity starts
where the run started, and a position held at its start counts as entered
there. The candidate with the best score wins the fold. NaN scores lose, ties
go to the lowest column, and a fold where every score is NaN (all candidates
still in their warmup, for example) stays flat with choice -1.

The out-of-sample target holds the chosen candidate's target in each test
window and 0 before the first one. It runs through the engine once, so a
position carried from one fold into the next pays no extra cost, and a switch
pays exactly the trade it causes.

With a portfolio engine, each candidate is a `(T, K)` target for the whole
book. Every window is scored on the book: equity, pnl and cost summed across
the assets, as [`position_test`](position_test.md) scores a portfolio. The
result's `target` is `(T, K)` and `output` `(T, K, 4)`, a callable metric gets
one candidate's `(L, K, 4)` window and returns one number, and `carry` is a
`(T, K)` rate, one column per asset.

A portfolio engine reads a finite target as an order and `NaN` as "hold", so a
rebalancing schedule has `NaN` between its rebalance steps. So that each test
window starts from the chosen candidate's book, a `NaN` on the first row of a
test window is restated as that candidate's last finite target for the asset,
or 0 when it has none (the flat book its own run starts from). The other rows
are the candidate's targets as given.

The function is batch-only. The backtests and the statistics run in C++; the
function only slices, compares, and stitches.

<!-- HELP_END -->

## Signature

`walk_forward(engine, targets, *prices, train, test, metric="sharpe",
anchored=False, periods_per_year=1.0, larger_is_better=None, carry=None,
carry_multiplier=1.0)`

- `engine`: a target-position engine, such as
  [`BacktestOHLCTarget`](BacktestOHLCTarget.md) or
  [`BacktestPriceTarget`](BacktestPriceTarget.md), or a portfolio engine,
  [`BacktestPortfolioTarget`](BacktestPortfolioTarget.md) or
  [`BacktestPortfolioOHLCTarget`](BacktestPortfolioOHLCTarget.md).
- `targets`: a `(T, K)` array with one target series per candidate, for
  example one column per parameter value. A `(T,)` series is one candidate.
  For a portfolio engine, `(T, K, C)`: C candidates for K assets (`(T, K)` is
  one candidate), with `(T, K)` prices.
- `prices`: the engine's market inputs, as in `engine(target, *prices)`.
- `train`: the length of each training window, at least 2 and shorter than
  the series.
- `test`: the length of each test window, at least 1. The last one can be
  shorter.
- `metric`: the selection statistic, any name from the
  [`backtest_report`](backtest_report.md) summary, or a function that maps an
  engine output of shape `(L, k, 4)` to `k` values. The default is the Sharpe
  ratio.
- `anchored`: train on all history before each test window instead of the
  last `train` steps.
- `periods_per_year`: annualizes the ratios, as in `backtest_report`. The
  default 1 gives per-step values.
- `larger_is_better`: the direction of the metric, as in
  [`position_test`](position_test.md). Statistics with no natural direction,
  such as `turnover`, need it.
- `carry`: a per-step carry rate (perpetual funding, borrow fee), a `(T,)`
  series (`(T, K)` for a portfolio engine) with the sign and timing
  [`BacktestCarry`](BacktestCarry.md) documents. With it, every candidate's
  run and the stitched run pass through
  `BacktestCarry(multiplier=carry_multiplier)` before they are scored, marked
  with the market input named by the engine's `mark_input` attribute:

  | Engine | Marked with |
  |:-------|:------------|
  | `BacktestPriceTarget` | the price |
  | `BacktestOHLCTarget`, `BacktestOHLCVTarget` | the close |
  | `BacktestTradesTarget` | the trade price |
  | `BacktestL1Target` | the ask |
  | `BacktestPortfolioTarget` | the price |
  | `BacktestPortfolioOHLCTarget` | the close |

  A subclass of one of these engines inherits its mark, and an engine without
  `mark_input` raises with `carry`. `output` and `stats` are after carry.
  `carry_multiplier` is the contract multiplier, as in `BacktestCarry`.

It returns a named tuple with these fields:

- `target`: the stitched out-of-sample target, shape `(T,)`, or `(T, K)` for a
  portfolio engine.
- `choice`: the column chosen in each fold, shape `(n_folds,)`.
- `fold_start`: the first step `t0` of each test window.
- `train_scores`: every candidate's metric on every training window, shape
  `(n_folds, K)`.
- `output`: the engine output on `target`, shape `(T, 4)` (`(T, K, 4)` for a
  portfolio engine), after carry when `carry` is given.
- `stats`: every `backtest_report` summary statistic of `output` from the
  first test window to the end, as a dict of floats.
- `metric`: the name of the metric.

## Choosing the windows

The training window has to hold enough trades for the metric to separate the
candidates, and the test window sets how often the choice is revised. The two
lengths are parameters of the procedure too: trying several and keeping the
best brings back the selection bias that walk-forward removes. Pick them from
the strategy's horizon, before looking at the result.

## Example

A trend rule on daily BTC with six lookback spans, a rolling one-year training
window, and a three-month test window. The
[walk-forward recipe](../recipes/walk-forward.md) plots the result and
compares it with the span that looks best in hindsight.

```{eval-rst}
.. exec_code::

    import numpy as np
    import pandas as pd
    from screamer import EwMean, LogReturn, BacktestOHLCTarget, walk_forward

    df = pd.read_csv("notebooks/data/binance_daily_20.csv")
    btc = df[df.symbol == "BTC"]
    o, h, l, c = (btc[k].to_numpy() for k in ["open", "high", "low", "close"])

    spans = [5, 10, 20, 40, 80, 160]
    ret = LogReturn()(c)
    X = np.column_stack([np.sign(np.nan_to_num(EwMean(span=s)(ret))) for s in spans])

    wf = walk_forward(BacktestOHLCTarget(taker_fee=5e-4), X, o, h, l, c,
                      train=365, test=91, periods_per_year=365)
    print("span per fold:", [spans[k] for k in wf.choice])
    print(f"out-of-sample Sharpe {wf.stats['sharpe']:.2f}, "
          f"max drawdown {wf.stats['max_drawdown']:,.0f} USD")
```

## References

- Pardo, R. (2008). *The Evaluation and Optimization of Trading Strategies*,
  2nd ed. Wiley.
- Bailey, D. H., Borwein, J., López de Prado, M., and Zhu, Q. J. (2014).
  Pseudo-mathematics and financial charlatanism: the effects of backtest
  overfitting on out-of-sample performance. *Notices of the AMS*, 61(5),
  458-471.
