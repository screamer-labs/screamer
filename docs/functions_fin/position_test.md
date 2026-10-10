---
name: position_test
title: position_test
kind: function
short: Test whether a backtest's timing beats randomly timed copies of the same strategy,
  with a p-value.
tags:
- p-value
- randomization test
- permutation test
- monte carlo
- significance test
- data snooping
see_also:
- strategy_test
- circular_shift
- segment_shuffle
- deflated_sharpe
references:
- Aronson, D. (2006). *Evidence-Based Technical Analysis*. Wiley.
- Masters, T. (2020). *Permutation and Randomization Tests for Trading System Development*.
  Independently published.
- White, H. (2000). A reality check for data snooping. *Econometrica*, 68(5), 1097-1126.
- Phipson, B. and Smyth, G. K. (2010). Permutation p-values should never be zero. *Statistical
  Applications in Genetics and Molecular Biology*, 9(1).
topics:
- significance
---

# `position_test`

Tests whether a backtest result could have come from luck. It keeps the market
data and the strategy's trading profile, randomizes **when** the strategy holds
its positions, replays every randomized copy through the same backtest engine,
and reports where the real result falls in that null distribution.

The null hypothesis is that the timing carries no information about future
returns, given this trading profile. The randomized copies keep the trades
themselves (their number, sides, sizes, and holding periods), so a long-biased
strategy in a rising market is compared with equally long-biased copies. The
test therefore measures timing skill beyond exposure. This is the Monte Carlo
permutation test for a fixed system (Aronson 2006; Masters 2020).

Two ways to randomize the timing:

- `method="shift"` (default) rotates the whole target series by a random
  offset (see [`circular_shift`](circular_shift.md)). Every property of the
  series is kept, including its autocorrelation.
- `method="segments"` shuffles the order of the runs: each trade, and each flat
  stretch, moves as one block (see [`segment_shuffle`](segment_shuffle.md)).

`metric` is the name of any statistic in the
[`backtest_report`](backtest_report.md) summary (`"sharpe"`, per step, is the
default; `"sortino"`, `"max_drawdown"`, `"win_rate"`, and so on), or a function
that maps an engine output of shape `(T, k, 4)` to `k` values. The p-value is the share of randomized copies that
do at least as well as the real result, counting the real result itself, so it
is never zero: with `n=999` the smallest possible p-value is 0.001.

The test is batch-only: it needs the whole series. The surrogates and the
backtests run in C++, and all randomized copies go through the engine as one
`(T, n)` call, processed in column chunks to bound memory.

<!-- HELP_END -->

## Signature

`position_test(engine, target, *market, n=1000, method="shift", metric="sharpe",
larger_is_better=None, min_shift=None, seed=0, periods_per_year=1.0)`

- `engine`: a target-position engine, such as
  [`BacktestOHLCTarget`](BacktestOHLCTarget.md),
  [`BacktestPriceTarget`](BacktestPriceTarget.md),
  [`BacktestTradesTarget`](BacktestTradesTarget.md), or
  [`BacktestL1Target`](BacktestL1Target.md), or a portfolio engine (see
  [A portfolio](#a-portfolio)). Order-posting engines are not
  supported: their limit prices are tied to the market at each moment, so
  moving them in time is not meaningful.
- `target`, `market`: the engine's inputs, exactly as in
  `engine(target, *market)`.
- `n`: the number of randomized copies.
- `larger_is_better`: the direction of the test. The default follows the
  metric: True for return and risk-adjusted statistics and for
  `max_drawdown` and `avg_loss` (both `<= 0`), False for `total_cost`, and
  True for a function. Statistics with no natural direction (`turnover`,
  `num_trades`, `exposure`, `round_trips`, `avg_holding`) need it.
- `min_shift`: the smallest rotation for `method="shift"`. The default is
  twice the mean run length, so no copy stays nearly aligned with the original.
- `seed`: the same seed gives the same copies on every platform, Python and
  JavaScript alike.
- `periods_per_year`: annualizes the ratios, as in `backtest_report`. The
  default 1 gives per-step values.

It returns a named tuple with these fields:

- `observed`: the metric of the real backtest.
- `null`: the metric of each randomized copy, NaN values left out.
- `p_value`, `metric`, `method`, `min_shift`.
- `family_p_value`: None for a single strategy (see below).
- `observed_stats`: every `backtest_report` summary statistic of the real
  backtest, as a dict of floats.
- `null_stats`: the same statistics for each randomized copy, as a dict of
  arrays of length `n`. They are in run order with NaN kept, so entry `j` of
  every array belongs to the same copy, and two statistics can be plotted
  against each other.

## Distributions of every statistic

The p-value covers one metric, but `null_stats` holds the full set, so you can
see where the real backtest falls on any statistic, or on two at once. Wrap
the dict in `pandas.DataFrame(res.null_stats)` for a table with one row per
randomized copy. The second plot in the [examples](#examples) shows two such
scatter plots.


## Several strategies

If you tried several variants and report the best one, test them together: pass
a `(T, K)` target with one column per variant (and `method="shift"`). Each
randomized round shifts all `K` columns by the same offset. The result then has
one `observed` and `p_value` per variant, a `null` of shape `(n, K)`, and
`family_p_value`, which compares each variant with the best of the `K`
randomized results in every round (the max statistic of White's Reality Check).
In `observed_stats` and `null_stats`, each statistic then has shape `(K,)` and
`(n, K)`.
The best variant's ordinary p-value is too optimistic, because picking the best
of many is itself a source of luck; its `family_p_value` is not.

## A portfolio

With a portfolio engine,
[`BacktestPortfolioTarget`](BacktestPortfolioTarget.md) or
[`BacktestPortfolioOHLCTarget`](BacktestPortfolioOHLCTarget.md), the test
treats the `(T, K)` target as one book of `K` assets, not as `K` separate
strategies. Every randomized round shifts all `K` assets by the same offset,
which keeps the book's composition and the relations between its assets, and
scores the whole book. The result is a single test: `observed`, `null`, and
`p_value` are those of the book, and `family_p_value` is None.

The book's statistics sum the assets: equity, pnl, and cost add up, turnover
and the trade count add each asset's position changes, and the trade
statistics (`round_trips`, `win_rate`, ...) count a round trip from a flat book
to a flat book. For a callable `metric`, the function receives one run's
`(T, K, 4)` output and returns one value.

## What it does not cover

The test checks one fixed target series. If the strategy was fitted, tuned, or
selected on the same data, that choice also inflates the result, and this test
does not see it. [`strategy_test`](strategy_test.md) re-runs the whole strategy
on permuted market data and covers it.

## Examples

The same trend-following signal runs on two markets: one whose returns have
mild momentum, and a random walk. The test separates them.

```{eval-rst}
.. plotly::
    :include-source: True

    import numpy as np
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from screamer import BacktestOHLCTarget, EwMean, position_test

    def market(phi, n=2000, seed=1):
        # returns with autocorrelation phi (phi = 0 is a random walk)
        e = np.random.default_rng(seed).standard_normal(n) * 0.01
        r = np.zeros(n)
        for t in range(1, n):
            r[t] = phi * r[t - 1] + e[t]
        close = 100 * np.exp(np.cumsum(r))
        open_ = np.r_[close[0], close[:-1]]
        return r, open_, np.maximum(open_, close) * 1.001, np.minimum(open_, close) * 0.999, close

    fig = make_subplots(rows=1, cols=2, subplot_titles=('momentum market', 'random walk'))
    for col, phi in enumerate([0.15, 0.0], start=1):
        r, o, h, l, c = market(phi)
        signal = np.sign(np.nan_to_num(EwMean(span=10)(r)))     # trend-following
        res = position_test(BacktestOHLCTarget(taker_fee=1e-4), signal, o, h, l, c, n=999)
        fig.add_trace(go.Histogram(x=res.null, nbinsx=40, marker_color='lightsteelblue',
                                   name='randomized timing', showlegend=col == 1), row=1, col=col)
        fig.add_vline(x=res.observed, line_color='crimson', line_width=3, row=1, col=col,
                      annotation_text=f'actual, p = {res.p_value:.3f}')
    fig.update_layout(title='position_test: Sharpe of the actual timing against 999 random timings',
                      xaxis_title='Sharpe per bar', xaxis2_title='Sharpe per bar',
                      margin=dict(l=20, r=20, t=80, b=20),
                      legend=dict(orientation='h', yanchor='bottom', y=1.08, xanchor='right', x=1))
    fig.show()

The same backtest, with two statistics of each randomized copy plotted against
each other and the actual backtest marked.

.. plotly::
    :include-source: True

    import numpy as np
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from screamer import BacktestOHLCTarget, EwMean, position_test

    rng = np.random.default_rng(1)
    e = rng.standard_normal(2000) * 0.01
    r = np.zeros(2000)
    for t in range(1, 2000):
        r[t] = 0.15 * r[t - 1] + e[t]                       # returns with momentum
    close = 100 * np.exp(np.cumsum(r))
    open_ = np.r_[close[0], close[:-1]]
    high, low = np.maximum(open_, close) * 1.001, np.minimum(open_, close) * 0.999
    signal = np.sign(np.nan_to_num(EwMean(span=10)(r)))

    res = position_test(BacktestOHLCTarget(taker_fee=1e-4), signal, open_, high, low, close,
                        n=999, periods_per_year=252)
    null, real = res.null_stats, res.observed_stats

    pairs = [('sharpe', 'max_drawdown'), ('win_rate', 'profit_factor')]
    fig = make_subplots(rows=1, cols=2, subplot_titles=[f'{x} vs {y}' for x, y in pairs])
    for col, (x, y) in enumerate(pairs, start=1):
        fig.add_trace(go.Scatter(x=null[x], y=null[y], mode='markers', name='randomized timing',
                                 marker=dict(size=4, color='lightsteelblue'),
                                 showlegend=col == 1), row=1, col=col)
        fig.add_trace(go.Scatter(x=[real[x]], y=[real[y]], mode='markers', name='actual',
                                 marker=dict(size=14, color='crimson', symbol='star'),
                                 showlegend=col == 1), row=1, col=col)
        fig.update_xaxes(title_text=x, row=1, col=col)
        fig.update_yaxes(title_text=y, row=1, col=col)
    fig.update_layout(title='Every statistic of 999 random timings, with the actual backtest',
                      margin=dict(l=20, r=20, t=80, b=20),
                      legend=dict(orientation='h', yanchor='bottom', y=1.08, xanchor='right', x=1))
    fig.show()
```
