---
name: CorwinSchultzSpread
title: Corwin-Schultz High-Low Spread
implementation_family: micro
topics:
- price-impact
tags:
- corwin-schultz
- high-low spread
- bid-ask spread
- spread estimator
- liquidity
- microstructure
- hlc
short: 'Corwin-Schultz (2012) bid-ask spread from the high and low of two consecutive bars,
  averaged over a trailing window.'
inputs: 3
outputs: 1
parameters:
- name: window_size
  type: int
  default: 20
  min: 1
  description: Number of most recent two-bar estimates averaged.
- name: adjust_overnight
  type: bool
  default: true
  description: Shift a bar's high and low by its gap from the previous close before pairing
    the two bars (the paper's overnight adjustment).
- name: start_policy
  type: str
  default: strict
  enum:
  - strict
  - expanding
  - zero
  description: 'Output before the window is full: "strict" gives NaN, "expanding" averages
    the two-bar estimates seen so far, and "zero" treats the window as pre-filled with zeros.'
nan_policy: ignore
see_also:
- AbdiRanaldoSpread
- RollSpread
- EffectiveSpread
- RollingParkinsonVar
references:
- 'Corwin, S. A. and Schultz, P. (2012). A simple way to estimate bid-ask spreads from daily
  high and low prices. *Journal of Finance*, 67(2), 719-760. doi:10.1111/j.1540-6261.2012.01729.x'
---

# `CorwinSchultzSpread`

## Description

`CorwinSchultzSpread` is the Corwin and Schultz (2012) estimate of the
proportional bid-ask spread from bar highs and lows. A bar's high is usually a
buy at the ask and its low a sell at the bid, so the high-low range holds both
the price variance and the spread. The variance part of the range grows with
the length of the period and the spread part does not, so comparing the range of
one bar with the range of two bars separates them.

For the pair of consecutive bars `t-1` and `t`, with `H` the high and `L` the low,

$$
\beta = \ln\!\left(\frac{H_{t-1}}{L_{t-1}}\right)^2 + \ln\!\left(\frac{H_t}{L_t}\right)^2,
\qquad
\gamma = \ln\!\left(\frac{\max(H_{t-1}, H_t)}{\min(L_{t-1}, L_t)}\right)^2,
$$

$$
\alpha = \frac{\sqrt{2\beta} - \sqrt{\beta}}{3 - 2\sqrt{2}} - \sqrt{\frac{\gamma}{3 - 2\sqrt{2}}},
\qquad
S = \frac{2\,(e^{\alpha} - 1)}{1 + e^{\alpha}}.
$$

A negative two-bar `S` is set to 0, and the output at bar `t` is the mean of
the last `window_size` two-bar estimates, the pair `(t-1, t)` included. The
paper averages over a month of daily bars. `S` is a fraction of the price: 0.01
is a spread of 1%.

With `adjust_overnight=True`, the gap between bars is taken out before the pair
is formed, as in the paper: if bar `t`'s low is above the close of bar `t-1`,
bar `t`'s high and low are both lowered by the difference, and if its high is
below that close they are both raised by it. Without this, a price move between
the bars counts as range and raises the estimate. Where each bar opens at the
previous close (24-hour markets, intraday bars), the adjustment rarely applies.

The `close` input is used only by the overnight adjustment, and a `NaN` close
still makes the bar unusable when the adjustment is off.

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
    from screamer import CorwinSchultzSpread

    rng = np.random.default_rng(3)
    n_bars, m = 600, 50                                       # 50 trades per bar
    spread = np.where(np.arange(n_bars) < 300, 0.002, 0.005)   # 20 bp, then 50 bp
    mid = 100 * np.exp(np.cumsum(rng.normal(0, 0.0005, n_bars * m)))
    side = rng.choice([-1.0, 1.0], n_bars * m)                # sell at bid, buy at ask
    price = (mid * (1 + side * np.repeat(spread, m) / 2)).reshape(n_bars, m)
    high, low, close = price.max(axis=1), price.min(axis=1), price[:, -1]

    cs = CorwinSchultzSpread(50)(high, low, close)

    fig = go.Figure()
    fig.add_trace(go.Scatter(y=spread, name='true spread', line=dict(color='gray', dash='dash')))
    fig.add_trace(go.Scatter(y=cs, name='Corwin-Schultz, 50 bars', line=dict(color='navy')))
    fig.update_layout(title='CorwinSchultzSpread: the spread from bar highs and lows',
                      xaxis_title='bar', yaxis_title='spread (fraction of price)',
                      margin=dict(l=20, r=20, t=60, b=20),
                      legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1))
    fig.show()
```

<!-- HELP_END -->
