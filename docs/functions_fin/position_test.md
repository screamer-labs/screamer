---
name: position_test
title: position_test
kind: function
short: Test whether a backtest's timing beats randomly timed copies of the same strategy, with a p-value.
topics:
- backtesting
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

`metric` is `"sharpe"` (per step, the default), `"total_pnl"`,
`"max_drawdown"`, or a function that maps an engine output of shape
`(T, k, 4)` to `k` values. The p-value is the share of randomized copies that
do at least as well as the real result, counting the real result itself, so it
is never zero: with `n=999` the smallest possible p-value is 0.001.

The test is batch-only: it needs the whole series. The surrogates and the
backtests run in C++, and all randomized copies go through the engine as one
`(T, n)` call, processed in column chunks to bound memory.

<!-- HELP_END -->

## Signature

`position_test(engine, target, *market, n=1000, method="shift", metric="sharpe",
larger_is_better=None, min_shift=None, seed=0)`

- `engine`: a target-position engine, such as
  [`BacktestOHLCTarget`](BacktestOHLCTarget.md),
  [`BacktestPriceTarget`](BacktestPriceTarget.md),
  [`BacktestTradesTarget`](BacktestTradesTarget.md), or
  [`BacktestL1Target`](BacktestL1Target.md). Order-posting engines are not
  supported: their limit prices are tied to the market at each moment, so
  moving them in time is not meaningful.
- `target`, `market`: the engine's inputs, exactly as in
  `engine(target, *market)`.
- `n`: the number of randomized copies.
- `larger_is_better`: the direction of the test. The default follows the
  metric (True for all three named metrics) and is True for a function.
- `min_shift`: the smallest rotation for `method="shift"`. The default is
  twice the mean run length, so no copy stays nearly aligned with the original.
- `seed`: the same seed gives the same copies on every platform, Python and
  JavaScript alike.

It returns a named tuple with `observed`, `null` (the metric of each copy,
NaN values left out), `p_value`, `metric`, `method`, and `min_shift`.

## What it does not cover

The test checks one fixed target series. If you tried many strategy variants
and kept the best one, test all of them and compare the best result with the
best result of each randomized round; otherwise the selection itself inflates
the significance.

## Example

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
```

## References

- Aronson, D. (2006). *Evidence-Based Technical Analysis*. Wiley.
- Masters, T. (2020). *Permutation and Randomization Tests for Trading System
  Development*. Independently published.
- Phipson, B. and Smyth, G. K. (2010). Permutation p-values should never be
  zero. *Statistical Applications in Genetics and Molecular Biology*, 9(1).
