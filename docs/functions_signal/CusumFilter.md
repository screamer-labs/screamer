---
name: CusumFilter
title: CUSUM event filter
implementation_family: signal
topics:
- signals
- regime
tags:
- cusum
- event filter
- event sampling
- structural break
- change detection
- labeling
short: 'Symmetric CUSUM event filter: +1 when the cumulative rise since the last event passes
  a threshold, -1 for a fall, 0 otherwise.'
inputs: 1
outputs: 1
parameters:
- name: threshold
  type: float
  default: 1.0
  required: true
  description: Cumulative move, in the units of the input, that triggers an event. Must be
    positive.
- name: drift
  type: float
  default: 0.0
  min: 0.0
  description: Amount subtracted from each rise and from each fall before it is accumulated,
    so moves smaller than `drift` per step do not build up. 0 or more.
nan_policy: ignore
see_also:
- Hold
- TripleBarrier
- BayesianChangepoint
- SchmittTrigger
references:
- 'Lopez de Prado, M. (2018). *Advances in Financial Machine Learning*, section 2.5.2.1. Wiley.'
---

# `CusumFilter`

## Description

`CusumFilter` is the symmetric CUSUM event filter (Lopez de Prado 2018). It
keeps two running sums of the input,

$$
S^+_t = \max(0,\ S^+_{t-1} + x_t - \text{drift}), \qquad
S^-_t = \min(0,\ S^-_{t-1} + x_t + \text{drift}),
$$

and outputs `-1` when `S- < -threshold`, `+1` when `S+ > threshold`, and `0`
otherwise. After an event the sum that fired restarts from 0, as in the code of
the book (section 2.5.2.1). At the step an event fires, the other sum is
already 0, so resetting it would change nothing. The input is normally
returns or price differences, so an event marks a cumulative move of more than
`threshold` since the last event, in either direction.

The events are a sampling rule: label or trade only at the steps where the
output is not 0, for example with
[`TripleBarrier`](../functions_fin/TripleBarrier.md) entries at those steps.
To hold a position for a while after each event, follow it with
[`Hold`](Hold.md).

With `threshold=1`, the input `[0.5, 0.6, -0.2, 0.3, -1.5]` gives
`[0, 1, 0, 0, -1]`: the sum of the first two values passes 1, and after the
reset the fall to -1.5 passes -1.

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
    from screamer import CusumFilter, Diff

    rng = np.random.default_rng(5)
    price = 100 + np.cumsum(rng.standard_normal(400) * 0.5)
    events = np.asarray(CusumFilter(threshold=2.0)(Diff(1)(price)))
    t = np.arange(len(price))

    fig = go.Figure()
    fig.add_trace(go.Scatter(y=price, name='price', line=dict(color='gray')))
    for value, color, name in [(1, 'seagreen', 'rise event (+1)'), (-1, 'crimson', 'fall event (-1)')]:
        at = t[events == value]
        fig.add_trace(go.Scatter(x=at, y=price[at], mode='markers', name=name,
                                 marker=dict(color=color, size=7)))
    fig.update_layout(title='CusumFilter: an event at each cumulative move of more than 2',
                      xaxis_title='step', yaxis_title='price',
                      margin=dict(l=20, r=20, t=60, b=20),
                      legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1))
    fig.show()
```

<!-- HELP_END -->
