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
short: Q3 minus Q1 over the trailing window.
inputs: 1
outputs: 1
parameters:
- name: window_size
  type: int
  default: 20
  min: 2
  description: Trailing-window length.
nan_policy: ignore
---

# `RollingIqr`

## Description

`RollingIqr` computes the rolling **inter-quartile range**:

$$
\text{IQR}[t] = Q_{0.75}[t] - Q_{0.25}[t]
$$

A robust spread measure: discards the top and bottom 25% of the window, so it is unaffected by single-point outliers. Useful as the denominator of a robust z-score, or for outlier-resistant volatility heuristics.

*Parameters*: `window_size` (int, positive).

<!-- NAN_FOOTNOTE_START -->
## NaN handling

**Policy: `ignore`.** A `NaN` in any input at index `t` causes the function to skip that step: output at `t` is `NaN` and internal state is unchanged. Subsequent finite samples are processed as if step `t` had not occurred.
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

A single `OrderStatisticTree` (the same primitive `RollingQuantile` uses) is queried twice per step -- once at the `0.25 * (n - 1)` position and once at `0.75 * (n - 1)` -- with linear interpolation between adjacent order statistics, identical to `RollingQuantile`'s formula.

### Why not just two `RollingQuantile` instances?

Composing as `RollingQuantile(w, 0.75)(x) - RollingQuantile(w, 0.25)(x)` would work but use two independent OSTs. The dedicated implementation has:

| | Two `RollingQuantile` | `RollingIqr` |
|---|---|---|
| Memory | 2 trees (≈ 2W nodes) | 1 tree (≈ W nodes) |
| Inserts/erases per step | 2 + 2 | 1 + 1 |
| Asymptotic complexity | `O(log W)` | `O(log W)` |

Same asymptotic complexity, **half the memory and half the work** per step. Validated in tests against the composition reference (post-warmup) to floating-point precision.

### Complexity

* Time complexity: `O(log W)` per step.
* Space complexity: `O(window_size)`.

## Reference

Equivalent to `pandas.Series.rolling(w).quantile(0.75) - pandas.Series.rolling(w).quantile(0.25)`.
