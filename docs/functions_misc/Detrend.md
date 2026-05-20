---
name: Detrend
title: Detrend
implementation_family: misc
topics:
- transforms
tags:
- detrend
- centered
short: x[t] minus its rolling mean.
inputs: 1
outputs: 1
parameters:
- name: window_size
  type: int
  default: 20
  min: 2
  description: Window for the trailing rolling mean.
- name: start_policy
  type: str
  default: strict
  enum:
  - strict
  - expanding
  - zero
  description: 'Warmup behaviour: ''strict'' (NaN until full window), ''expanding''
    (use partial windows), or ''zero'' (treat missing as zero).'
---

# `Detrend`

## Description

The `Detrend` function removes a slow-moving trend from a series by subtracting a rolling mean from the input. The result emphasises the deviation of each sample from its recent average and is useful for centring an input before another transform, or for visualising short-term fluctuations on top of a drifting baseline.

*Equation*:

$$
y[t] = x[t] - \frac{1}{w} \sum_{k=0}^{w-1} x[t-k]
$$

where $w$ is the `window_size`.

*Parameters*:

- `window_size` (int): width of the rolling-mean window in samples.
- `start_policy` (str, optional): warmup behaviour, `"strict"` (default), `"truncate"`, or `"zero"`. Same semantics as the rolling functions.

*NaN handling*: Under `start_policy="strict"` the rolling mean is NaN until `window_size` samples have been observed, and `Detrend` therefore emits NaN during that warmup. NaN inputs propagate through the rolling buffer.

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

`Detrend` wraps a single internal `RollingMean` buffer. On each call it appends the new sample to the buffer and returns `x[t] - mean[t]`. There is no extra storage beyond the rolling-mean window.
