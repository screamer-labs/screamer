---
name: EdgeSpread
title: EDGE Bid-Ask Spread
implementation_family: micro
topics:
- price-impact
tags:
- edge
- ardia-guidotti-kroencke
- bid-ask spread
- spread estimator
- liquidity
- microstructure
- ohlc
- bid-ask spread estimator
- effective spread from ohlc
short: 'EDGE bid-ask spread of Ardia, Guidotti and Kroencke (2024) from the open, high,
  low and close of the bars in a trailing window.'
inputs: 4
outputs: 1
parameters:
- name: window_size
  type: int
  default: 20
  min: 3
  description: Number of most recent bars in the estimate.
- name: sign
  type: bool
  default: false
  description: Return a negative value when the estimated squared spread is negative. With
    false the output is the square root of its absolute value.
- name: start_policy
  type: str
  default: strict
  enum:
  - strict
  - expanding
  description: 'Output before the window is full: "strict" gives NaN and "expanding" estimates
    from the bars seen so far.'
nan_policy: nan-aware
see_also:
- CorwinSchultzSpread
- AbdiRanaldoSpread
- RollSpread
- EffectiveSpread
references:
- 'Ardia, D., Guidotti, E. and Kroencke, T. A. (2024). Efficient estimation of bid-ask spreads
  from open, high, low, and close prices. *Journal of Financial Economics*, 161, 103916.
  doi:10.1016/j.jfineco.2024.103916'
---

# `EdgeSpread`

## Description

`EdgeSpread` is the EDGE estimate of the proportional bid-ask spread of
Ardia, Guidotti and Kroencke (2024), from the open, high, low and close prices
of the bars in a trailing window. The open, high, low and close are each
assumed to be a trade at the bid or the ask, and the estimator combines the
returns between them in pairs of consecutive bars into two unbiased estimates of
the squared spread, weighted by their variances to minimize the estimation
error. Under the paper's model its estimate of the squared spread is unbiased
whatever the ratio of spread to volatility, so on average it stays near the
spread where it is small next to the price moves within a bar, the regime in
which [`CorwinSchultzSpread`](CorwinSchultzSpread.md) and
[`AbdiRanaldoSpread`](AbdiRanaldoSpread.md) are biased.

For two consecutive bars, with `o`, `h`, `l`, `c` the log prices of the
second bar, `h1`, `l1`, `c1` those of the first, and `m = (h + l) / 2`,
`m1 = (h1 + l1) / 2` the log mid-ranges, the returns are

$$
r_1 = m - o,\quad r_2 = o - m_1,\quad r_3 = m - c_1,\quad r_4 = c_1 - m_1,\quad r_5 = o - c_1 .
$$

The indicator `tau` is 1 when the second bar trades at all (`h != l` or
`l != c1`), and with averages over the pairs of the window,
`pt = mean(tau)`, `po = mean(tau and o != h) + mean(tau and o != l)` and
`pc = mean(tau and c1 != h1) + mean(tau and c1 != l1)`. With the
de-meaned returns `d_k = r_k - tau mean(r_k) / pt` for `k = 1, 3, 5`,

$$
x_1 = -\frac{4}{p_o} d_1 r_2 - \frac{4}{p_c} d_3 r_4, \qquad
x_2 = -\frac{4}{p_o} d_1 r_5 - \frac{4}{p_c} d_5 r_4 ,
$$

and with `e_k` and `v_k` the mean and variance of `x_k` over the window, the
squared spread is

$$
S^2 = \frac{v_2 e_1 + v_1 e_2}{v_1 + v_2},
$$

or `(e_1 + e_2) / 2` when `v_1 + v_2` is not positive. The output is
`sqrt(|S^2|)`, and with `sign=True` it is negative when `S^2` is, which
happens when the estimation noise is larger than the squared spread. The
output is `NaN` when fewer than two pairs have `tau = 1` or when `po` or `pc` is
0, for example on bars that do not trade. `S` is a fraction of the price: 0.01
is a spread of 1%.

The output at bar `t` is the authors' reference estimator applied to bars
`t - window_size + 1` to `t`, with `window_size - 1` pairs. The paper estimates
monthly from daily bars. Each step adds one pair to the window sums and removes
one. The sums are recomputed once per window, and at once when an outlier bar
leaves the window, so no rounding builds up; O(1) per bar on average.

Missing values follow the authors' reference code. A price that is `NaN`, or
not positive and finite, is missing. Each indicator and return leaves out the
pairs in which a price it uses is missing, and each average is over the pairs
that are left, so a bar with a missing open still contributes its high, low and
close. A bar takes its place in the window whatever is missing, so the window
always spans `window_size` bars, and the output at a bar with missing prices is
the estimate from the window.

## Limitations

The estimate of `S^2` is unbiased, but its variance is large when the spread
is small next to the volatility of a bar: the estimate then scatters around a small
value and is often negative, and the default `sign=False` reports
`sqrt(|S^2|)` for those windows too. Use `sign=True` where the spread may be
small, and a long window (the paper uses a month of daily bars). The estimator
assumes that the open, high, low and close are trades on a continuous price;
bars of a few trades on a coarse tick grid, where the open is often the high or
the low, do not satisfy it. Where quotes are available,
[`EffectiveSpread`](EffectiveSpread.md) and
[`RealizedSpread`](RealizedSpread.md) measure the spread directly.

<!-- NAN_FOOTNOTE_START -->
## NaN handling

**Policy: `nan-aware`.** This function is built to process `NaN` inputs, as the Description explains. See [NaN and warmup](../nan_and_warmup.md).
<!-- NAN_FOOTNOTE_END -->

## Examples

### Usage plot

```{eval-rst}
.. plotly::
    :include-source: True

    import numpy as np
    import plotly.graph_objects as go
    from screamer import CorwinSchultzSpread, EdgeSpread

    rng = np.random.default_rng(3)
    n_bars, m = 600, 50                                       # 50 trades per bar
    spread = np.where(np.arange(n_bars) < 300, 0.001, 0.003)   # 10 bp, then 30 bp
    mid = 100 * np.exp(np.cumsum(rng.normal(0, 0.002, n_bars * m)))
    side = rng.choice([-1.0, 1.0], n_bars * m)                # sell at bid, buy at ask
    price = (mid * (1 + side * np.repeat(spread, m) / 2)).reshape(n_bars, m)
    open_, close = price[:, 0], price[:, -1]
    high, low = price.max(axis=1), price.min(axis=1)

    edge = EdgeSpread(100, sign=True)(open_, high, low, close)
    cs = CorwinSchultzSpread(100)(high, low, close)

    fig = go.Figure()
    fig.add_trace(go.Scatter(y=spread, name='true spread', line=dict(color='gray', dash='dash')))
    fig.add_trace(go.Scatter(y=edge, name='EDGE, 100 bars', line=dict(color='darkorange')))
    fig.add_trace(go.Scatter(y=cs, name='Corwin-Schultz, 100 bars', line=dict(color='navy', width=1)))
    fig.update_layout(title='EdgeSpread: the spread from open, high, low and close',
                      xaxis_title='bar', yaxis_title='spread (fraction of price)',
                      margin=dict(l=20, r=20, t=60, b=20),
                      legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1))
    fig.show()
```

<!-- HELP_END -->
