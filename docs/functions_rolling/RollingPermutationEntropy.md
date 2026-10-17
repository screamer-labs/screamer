---
name: RollingPermutationEntropy
title: Rolling Permutation Entropy
implementation_family: rolling
topics:
- statistics
- regime
tags:
- permutation entropy
- ordinal patterns
- bandt-pompe
- complexity
- randomness
- predictability
- entropy
short: Entropy of the ordinal patterns of a series over a trailing window, from 0 (one pattern
  repeats) to 1 (every pattern equally often).
inputs: 1
outputs: 1
parameters:
- name: window_size
  type: int
  default: 100
  min: 2
  description: Number of patterns, ending at the most recent steps, in the window.
- name: order
  type: int
  default: 3
  min: 2
  description: Number of values in a pattern, 2 to 7. There are order! patterns.
- name: delay
  type: int
  default: 1
  min: 1
  description: Steps between the values of a pattern. `(order - 1) * delay` must be 16777216
    or less.
- name: start_policy
  type: str
  default: strict
  enum:
  - strict
  - expanding
  - zero
  description: 'Output before the window holds window_size patterns: "strict" gives NaN,
    "expanding" uses the patterns seen so far, and "zero" behaves as if the input were preceded
    by zeros.'
nan_policy: ignore
see_also:
- RollingVarianceRatio
- RollingHurst
- RollingJumpShare
references:
- 'Bandt, C. and Pompe, B. (2002). Permutation entropy: a natural complexity measure for time
  series. *Physical Review Letters* 88(17), 174102.'
---

# `RollingPermutationEntropy`

## Description

`RollingPermutationEntropy` is the permutation entropy of Bandt and Pompe
(2002) over a trailing window. At each step, the `order` values
`x[t - (order-1)*delay], ..., x[t - delay], x[t]` form an ordinal pattern: the
order of their ranks, one of `order!` possibilities. The output is the Shannon
entropy of the patterns that end at the last `window_size` steps, divided by
`log(order!)`:

$$
H = -\frac{1}{\log(\text{order}!)} \sum_{\pi} p_\pi \log p_\pi
$$

where `p_pi` is the share of pattern `pi` in the window. `H` is 0 when one
pattern fills the window, such as on a monotone or constant series, and near 1
when every pattern is equally frequent, as for white noise. Values in between
mean that some orderings are more common than others, which a forecast can
use.

Equal values rank by order of appearance, the earlier one lower, so on prices
that repeat a tick, `(2, 1, 1)` and `(1, 1, 2)` are two patterns. The input can
be prices or returns; a pattern of prices describes the signs of the moves
between them.

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
    from screamer import RollingPermutationEntropy

    rng = np.random.default_rng(6)
    noise = rng.standard_normal(1500)
    cycle = np.sin(np.arange(1500) * 2 * np.pi / 25) + 0.2 * rng.standard_normal(1500)
    x = np.where(np.arange(1500) < 750, noise, cycle)     # random, then a cycle
    h = RollingPermutationEntropy(200, order=4)(x)

    fig = go.Figure()
    fig.add_trace(go.Scatter(y=h, name='entropy', line=dict(color='navy')))
    fig.update_layout(title='RollingPermutationEntropy: near 1 on noise, lower on a cycle',
                      xaxis_title='step', yaxis_title='normalized entropy',
                      margin=dict(l=20, r=20, t=60, b=20), showlegend=False)
    fig.show()
```

<!-- HELP_END -->
