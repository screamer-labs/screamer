---
name: RollingIqr
title: Rolling interquartile range
implementation_family: rolling
topics:
- statistics
tags:
- iqr
- quartile
- rolling
- interquartile range
- robust spread
short: Q3 minus Q1 over the trailing window.
inputs: 1
outputs: 1
parameters:
- name: window_size
  type: int
  default: 20
  min: 1
  description: Number of most recent samples whose quartiles are taken.
see_also:
- RollingQuantile
- RollingMedian
- RollingMedianAD
- RollingMad
nan_policy: ignore
---

# `RollingIqr`

## Description

`RollingIqr` computes the rolling interquartile range:

$$
\text{IQR}[t] = Q_{0.75}[t] - Q_{0.25}[t]
$$

It is a robust spread measure: it ignores the top and bottom quarter of the window, so
a single outlier does not change it. Use it as the denominator of a robust z-score or
as an outlier-resistant volatility measure.

<!-- NAN_FOOTNOTE_START -->
## NaN handling

**Policy: `ignore`.** A `NaN` in any input gives a `NaN` output at that step. The step is skipped: it does not enter the window or update the internal state. See [NaN and warmup](../nan_and_warmup.md).
<!-- NAN_FOOTNOTE_END -->

## Examples

### Usage example

```{eval-rst}
.. plotly::
    :include-source: True

    import numpy as np
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from screamer import RollingIqr

    rng = np.random.default_rng(0)
    N = 300
    data = np.cumsum(rng.standard_normal(N))

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.5, 0.5],
                        vertical_spacing=0.08)
    fig.add_trace(go.Scatter(y=data, mode='lines', name='Input', line=dict(color='steelblue', width=1)),
                  row=1, col=1)
    fig.add_trace(go.Scatter(y=RollingIqr(window_size=30)(data), mode='lines', name='RollingIqr(window_size=30)',
                             line=dict(color='crimson', width=2)), row=2, col=1)
    fig.update_layout(
        title="Rolling interquartile range over a random walk",
        yaxis=dict(title='Input'), yaxis2=dict(title='IQR'),
        margin=dict(l=20, r=20, t=60, b=20),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
    )
    fig.show()
```

<!-- HELP_END -->

## Implementation Details

### Algorithm

A single order-statistic tree (the structure [`RollingQuantile`](RollingQuantile.md)
uses) is queried twice per step, at positions `0.25 * (n - 1)` and `0.75 * (n - 1)`,
with linear interpolation between adjacent order statistics, the same formula as
`RollingQuantile`.

### Comparison with two `RollingQuantile` instances

`RollingQuantile(w, 0.75)(x) - RollingQuantile(w, 0.25)(x)` gives the same result
with two independent trees:

| | Two `RollingQuantile` | `RollingIqr` |
|---|---|---|
| Memory | 2 trees (≈ 2W nodes) | 1 tree (≈ W nodes) |
| Inserts/erases per step | 2 + 2 | 1 + 1 |
| Asymptotic complexity | `O(log W)` | `O(log W)` |

The asymptotic complexity is the same, with half the memory and half the tree updates
per step. Tests check `RollingIqr` against the composition after warmup to
floating-point precision.

### Complexity

* Time complexity: `O(log W)` per step.
* Space complexity: `O(window_size)`.

## Validation

Equivalent to `pandas.Series.rolling(w).quantile(0.75) - pandas.Series.rolling(w).quantile(0.25)`.
