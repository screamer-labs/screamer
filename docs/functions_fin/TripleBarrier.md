---
name: TripleBarrier
title: Triple-barrier label on a price series
implementation_family: fin
topics:
- supervised
tags:
- triple barrier
- label
- labeling
- target
- classification
- meta-labeling
- take profit
- stop loss
- vertical barrier
- machine learning
short: 'A three-state label over a window: +1 when the price reaches the upper barrier first,
  -1 when it reaches the lower barrier first, 0 when it reaches neither within `horizon`
  steps.'
inputs: 1
outputs: 3
parameters:
- name: horizon
  type: int
  default: 20
  min: 1
  description: Steps after each entry in which a barrier can be reached. It is also the vertical
    barrier, and the delay of the output.
- name: upper
  type: float
  default: 0.0
  min: 0.0
  description: Distance of the upper barrier above the entry price, in `barrier_unit`. 0 turns
    it off.
- name: lower
  type: float
  default: 0.0
  min: 0.0
  description: Distance of the lower barrier below the entry price, in `barrier_unit`. 0 turns
    it off. Below 1 with barrier_unit="fraction".
- name: barrier_unit
  type: str
  default: fraction
  description: 'Unit of `upper` and `lower`: "fraction" of the entry price (0.02 = 2%), "price"
    (an absolute amount), or "vol" (multiples of the volatility expected over the horizon).'
- name: vol_span
  type: float
  default: 100.0
  description: Span of the exponentially weighted standard deviation of one-step log returns
    that sets the "vol" barriers. Must be greater than 1. Used only with barrier_unit="vol".
nan_policy: nan-aware
see_also:
- TripleBarrierOHLC
- forecast_pairs
- BacktestOHLCTarget
- LogReturn
output_names:
- label
- return
- holding
references:
- 'Lopez de Prado, M. (2018). *Advances in Financial Machine Learning*, chapter 3. Wiley.'
---

# `TripleBarrier`

## Description

`TripleBarrier` is a three-state label over a window. Every step `e` is an
entry at its price `p_e`. Over the next `horizon` steps the label is `+1`
when the price reaches the upper barrier first, `-1` when it reaches the lower
barrier first, and `0` when it reaches neither (Lopez de Prado 2018). The
label describes the price path only; no position or side is involved.

It returns three columns:

0. `label`: `+1`, `-1`, or `0`.
1. `return`: `exit / p_e - 1`, the simple return from the entry price to the
   exit price. The exit price is the first price at or beyond a barrier, or
   the price `horizon` steps after entry when no barrier is reached.
2. `holding`: the steps from entry to exit, 1 to `horizon`.

**Stamping.** The output at step `t` is the entry at `t - horizon`, the first
step at which its outcome is known. The first `horizon` outputs are `NaN`.
Pair it with the features of its entry step with
[`forecast_pairs(X, y, count=horizon)`](../functions_supervised/forecast_pairs.md).

**Barriers.** `upper` is the distance above the entry price and `lower` the
distance below it, in `barrier_unit`:

- `"fraction"`: `p_e * (1 + upper)` and `p_e * (1 - lower)`;
- `"price"`: `p_e + upper` and `p_e - lower`;
- `"vol"`: `p_e * (1 + upper * s)` and `p_e * (1 - lower * s)`, where `s` is
  the `EwStd(span=vol_span)` of the one-step log returns up to and including
  `e`, times `sqrt(horizon)`.

The units match the backtest engines' `stop_loss` and `take_profit`, so the
same numbers mean the same levels. A barrier set to `0` is off. With both off,
every label is `0` and `return` is the return over the horizon. A price
exactly at a barrier reaches it.

**Missing prices.** A `NaN` or non-positive price at entry or inside an
entry's window makes that entry's three outputs `NaN`; other entries are
unaffected. With `"vol"`, a log return that involves such a price is skipped
by the volatility estimate, and an entry whose `s` is `NaN` or zero (before
the estimate has two returns, or on a constant price) is `NaN`.

**Meta-labeling.** With a rule that gives a `side` (`+1` long, `-1` short) at
each step, and symmetric barriers, the trade worked when `label == side`, and
its return is `side * return`.

[`TripleBarrierOHLC`](TripleBarrierOHLC.md) tests the barriers on each bar's
high and low instead of one price.

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
    from screamer import TripleBarrier

    rng = np.random.default_rng(3)
    n, h = 200, 10
    price = 100 * np.exp(np.cumsum(rng.standard_normal(n) * 0.01))

    out = TripleBarrier(horizon=h, upper=0.03, lower=0.03)(price)
    label = out[h:, 0]                       # row e: the label of the entry at step e
    steps = np.arange(n - h)

    fig = go.Figure()
    fig.add_trace(go.Scatter(y=price, name='price', line=dict(color='gray')))
    for value, color, name in [(1, 'seagreen', 'upper first (+1)'),
                               (-1, 'crimson', 'lower first (-1)'),
                               (0, 'steelblue', 'neither (0)')]:
        at = steps[label == value]
        fig.add_trace(go.Scatter(x=at, y=price[at], mode='markers', name=name,
                                 marker=dict(color=color, size=6)))
    fig.update_layout(title='TripleBarrier: each entry marked by the barrier its price reaches first',
                      yaxis=dict(title='price'), margin=dict(l=20, r=20, t=60, b=20),
                      legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1))
    fig.show()
```

<!-- HELP_END -->
