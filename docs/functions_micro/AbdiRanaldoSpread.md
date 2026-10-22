---
name: AbdiRanaldoSpread
title: Abdi-Ranaldo Close-High-Low Spread
implementation_family: micro
topics:
- price-impact
tags:
- abdi-ranaldo
- chl spread
- close-high-low
- bid-ask spread
- spread estimator
- liquidity
- microstructure
- hlc
short: 'Abdi-Ranaldo (2017) bid-ask spread from the close and the high-low midpoints of
  consecutive bars, over a trailing window.'
inputs: 3
outputs: 1
parameters:
- name: window_size
  type: int
  default: 20
  min: 1
  description: Number of most recent two-bar products averaged.
- name: start_policy
  type: str
  default: strict
  enum:
  - strict
  - expanding
  - zero
  description: 'Output before the window is full: "strict" gives NaN, "expanding" averages
    the two-bar products seen so far, and "zero" treats the window as pre-filled with zeros.'
nan_policy: ignore
see_also:
- CorwinSchultzSpread
- RollSpread
- EffectiveSpread
references:
- 'Abdi, F. and Ranaldo, A. (2017). A simple estimation of bid-ask spreads from daily close,
  high, and low prices. *Review of Financial Studies*, 30(12), 4437-4480. doi:10.1093/rfs/hhx084'
---

# `AbdiRanaldoSpread`

## Description

`AbdiRanaldoSpread` is the Abdi and Ranaldo (2017) estimate of the bid-ask
spread from bar close, high and low prices. The midpoint of a bar's high and low
is a proxy for the bar's efficient price, and a close that trades at the bid or
the ask sits half a spread away from it. The covariance of that distance with the
distance to the next bar's midpoint measures the squared spread.

With `c = ln(close)` and the log midpoint `eta = (ln(high) + ln(low)) / 2`, each
pair of consecutive bars `t-1` and `t` gives the product

$$
x_t = 4\,(c_{t-1} - \eta_{t-1})\,(c_{t-1} - \eta_t),
$$

and the output at bar `t` is

$$
S_t = \sqrt{\max\!\left(\frac{1}{n}\sum_{i=t-n+1}^{t} x_i,\; 0\right)}
$$

over the last `n = window_size` products. The products are averaged first and a
negative average is set to 0 before the square root, which is the paper's
monthly estimator. A single product can be negative, so a short window gives 0
more often. `S` is in log-price units, close to the spread as a fraction of the
price: a spread of 0.01 is about 1%.

A bar with a `NaN` input is skipped and left out of the pairs: the bars on either
side of it form a pair.

<!-- NAN_FOOTNOTE_START -->
## NaN handling

**Policy: `ignore`.** A `NaN` in any input gives a `NaN` output at that step. The step is skipped: it does not enter the window or update the internal state. See [NaN and warmup](../nan_and_warmup.md).
<!-- NAN_FOOTNOTE_END -->

## Examples

### Usage plot

```{eval-rst}
.. plotly::
    :include-source: True

    import numpy as np
    import plotly.graph_objects as go
    from screamer import AbdiRanaldoSpread, CorwinSchultzSpread

    rng = np.random.default_rng(3)
    n_bars, m = 600, 50                                       # 50 trades per bar
    spread = np.where(np.arange(n_bars) < 300, 0.002, 0.005)   # 20 bp, then 50 bp
    mid = 100 * np.exp(np.cumsum(rng.normal(0, 0.0005, n_bars * m)))
    side = rng.choice([-1.0, 1.0], n_bars * m)                # sell at bid, buy at ask
    price = (mid * (1 + side * np.repeat(spread, m) / 2)).reshape(n_bars, m)
    high, low, close = price.max(axis=1), price.min(axis=1), price[:, -1]

    ar = AbdiRanaldoSpread(50)(high, low, close)
    cs = CorwinSchultzSpread(50)(high, low, close)

    fig = go.Figure()
    fig.add_trace(go.Scatter(y=spread, name='true spread', line=dict(color='gray', dash='dash')))
    fig.add_trace(go.Scatter(y=ar, name='Abdi-Ranaldo, 50 bars', line=dict(color='crimson')))
    fig.add_trace(go.Scatter(y=cs, name='Corwin-Schultz, 50 bars', line=dict(color='navy', width=1)))
    fig.update_layout(title='AbdiRanaldoSpread: the spread from close, high and low',
                      xaxis_title='bar', yaxis_title='spread (fraction of price)',
                      margin=dict(l=20, r=20, t=60, b=20),
                      legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1))
    fig.show()
```

<!-- HELP_END -->
