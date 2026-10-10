---
name: strategy_test
title: strategy_test
kind: function
short: Test a whole strategy, model included, against the same strategy run on permuted market data or shifted features.
topics:
- backtesting
---

# `strategy_test`

Tests whether a strategy finds real structure in its data. It runs the whole
strategy once on the real inputs and `n` times on randomized copies, and reports
where the real result falls in that null distribution. The strategy is any
callable that takes the inputs positionally and returns a backtest engine's
`(T, 4)` output: a function built from screamer operators, a compiled `Pipeline`
(its `(values, index)` result is accepted), or your own code.

Because the whole strategy runs again on every copy, anything it fits, tunes,
or selects from the data is redone on the randomized data too. The test
therefore also accounts for that fitting, which
[`position_test`](position_test.md) cannot do: that test keeps one fixed
position series.

`method` chooses what is randomized:

- `"bars"` (default): the OHLC inputs at positions `ohlc` are replaced by
  bar-permuted copies (see [`permute_bars`](permute_bars.md)). The null
  hypothesis is that the market has no serial structure the strategy can use.
  This is the Monte Carlo permutation test of Masters (2020).
- `"returns"`: the single price input at position `price` is replaced by
  return-permuted copies (see [`permute_returns`](permute_returns.md)).
- `"shift"`: the feature inputs at positions `columns` are rotated in time by
  one random offset, the same for all of them, while the prices stay as they
  are. This asks whether those features carry information about the market.
  Their autocorrelation and their relations to each other are kept, so the
  model sees inputs from the same distribution. `min_shift` defaults to a tenth
  of the series.

With `"bars"` and `"returns"`, every other input moves with its bar, so per-bar
data such as volume stays with the bar it belongs to. Inputs that must not move,
such as timestamps, go in `fixed`.

`metric` is `"sharpe"` (per step, the default), `"total_pnl"`,
`"max_drawdown"`, or a function that maps an output of shape `(T, k, 4)` to `k`
values. The p-value counts the observed result itself, so it is never zero.

The test is batch-only. The permutations run in C++, and the cost is `n` runs of
the strategy; a strategy built from screamer operators typically runs in well
under a millisecond per thousand bars.

<!-- HELP_END -->

## Signature

`strategy_test(strategy, *inputs, n=500, method="bars", ohlc=(0, 1, 2, 3),
price=0, log=None, columns=None, fixed=(), metric="sharpe",
larger_is_better=None, min_shift=None, seed=0)`

It returns a named tuple with `observed`, `null` (NaN values left out),
`p_value`, `metric`, and `method`. `log` applies to `method="returns"`: log
returns or differences, by default log returns when every price is positive.
The same `seed` gives the same randomized inputs in Python and JavaScript.

## Example

A trend-following strategy on two markets: one whose returns have momentum, and
a random walk. Bar permutation removes the momentum, so only the first market
beats its permuted copies.

```{eval-rst}
.. plotly::
    :include-source: True

    import numpy as np
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from screamer import BacktestOHLCTarget, EwMean, LogReturn, strategy_test

    def market(phi, n=1500, seed=1):
        rng = np.random.default_rng(seed)
        e = rng.standard_normal(n) * 0.01
        r = np.zeros(n)
        for t in range(1, n):
            r[t] = phi * r[t - 1] + e[t]
        close = 100 * np.exp(np.cumsum(r))
        open_ = np.r_[close[0], close[:-1]] * np.exp(rng.standard_normal(n) * 0.001)
        return open_, np.maximum(open_, close) * 1.002, np.minimum(open_, close) * 0.998, close

    def strategy(o, h, l, c):
        signal = np.sign(np.nan_to_num(EwMean(span=10)(LogReturn()(c))))
        return BacktestOHLCTarget(taker_fee=1e-4)(signal, o, h, l, c)

    fig = make_subplots(rows=1, cols=2, subplot_titles=('momentum market', 'random walk'))
    for col, phi in enumerate([0.15, 0.0], start=1):
        res = strategy_test(strategy, *market(phi), n=999)
        fig.add_trace(go.Histogram(x=res.null, nbinsx=40, marker_color='lightsteelblue',
                                   name='permuted bars', showlegend=col == 1), row=1, col=col)
        fig.add_vline(x=res.observed, line_color='crimson', line_width=3, row=1, col=col,
                      annotation_text=f'actual, p = {res.p_value:.3f}')
    fig.update_layout(title='strategy_test: Sharpe on the real bars against 999 permuted markets',
                      xaxis_title='Sharpe per bar', xaxis2_title='Sharpe per bar',
                      margin=dict(l=20, r=20, t=80, b=20),
                      legend=dict(orientation='h', yanchor='bottom', y=1.08, xanchor='right', x=1))
    fig.show()
```

## References

- Masters, T. (2020). *Permutation and Randomization Tests for Trading System
  Development*. Independently published.
- Phipson, B. and Smyth, G. K. (2010). Permutation p-values should never be
  zero. *Statistical Applications in Genetics and Molecular Biology*, 9(1).
