---
name: Detrend
title: Detrend
implementation_family: misc
topics:
- standardization
- trend
tags:
- detrend
- centered
- demean
- remove trend
- rolling mean
short: x[t] minus its rolling mean.
inputs: 1
outputs: 1
parameters:
- name: window_size
  type: int
  default: 20
  min: 1
  description: Number of most recent samples in the trailing rolling mean that is subtracted.
- name: start_policy
  type: str
  default: strict
  enum:
  - strict
  - expanding
  - zero
  description: 'Output before the window is full: "strict" gives NaN, "expanding" computes
    from the samples seen so far, and "zero" treats the window as pre-filled with zeros.'
nan_policy: ignore
see_also:
- RollingMean
- RollingZscore
- Diff
- Decycler
---

# `Detrend`

## Description

The `Detrend` function removes a slow-moving trend from a series by subtracting a rolling mean from the input. The result is the deviation of each sample from its recent average. Use it to center an input before another transform, or to show short-term fluctuations around a drifting baseline.

*Equation*:

$$
y[t] = x[t] - \frac{1}{w} \sum_{k=0}^{w-1} x[t-k]
$$

where $w$ is the `window_size`.


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
    from screamer import Detrend

    rng = np.random.default_rng(11)
    n = 300
    trend = np.linspace(0, 5, n)
    seasonal = 0.5 * np.sin(np.linspace(0, 8 * np.pi, n))
    noise = rng.normal(0.0, 0.2, n)
    x = trend + seasonal + noise

    detrended = Detrend(30)(x)

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.5, 0.5], vertical_spacing=0.1)
    fig.add_trace(go.Scatter(y=x, mode='lines', name='Original (trend + season + noise)'),
                  row=1, col=1)
    fig.add_trace(go.Scatter(y=detrended, mode='lines',
                             name='Detrend(30)',
                             line=dict(color='green')),
                  row=2, col=1)
    fig.update_layout(
        title="Detrend: x[t] minus a 30-Sample Rolling Mean",
        xaxis_title="Index",
        yaxis_title="Original",
        yaxis2_title="Detrended",
        margin=dict(l=20, r=20, t=60, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    fig.show()
```

<!-- HELP_END -->

## Implementation Details

`Detrend` keeps one rolling-mean window. Each step adds the new sample to the window and returns `x[t] - mean[t]`, at O(1) cost. Memory is `O(window_size)`.
