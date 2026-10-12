---
name: RollingStd
title: Rolling standard deviation
implementation_family: rolling
topics:
- statistics
- volatility
tags:
- std
- volatility
- rolling
- realized volatility
- standard deviation
- rolling std
short: Trailing-window sample standard deviation (ddof=1).
inputs: 1
outputs: 1
parameters:
- name: window_size
  type: int
  default: 20
  min: 2
  description: Number of most recent samples in the window.
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
- RollingVar
- EwStd
- RollingZscore
- RollingParkinsonVol
- RollingYangZhangVol
---

# `RollingStd`

## Description

`RollingStd` computes the sample standard deviation (ddof=1) over the `window_size` most recent values:

$$
\text{std}_w[t] = \sqrt{\frac{1}{n-1} \sum_{i=t-w+1}^{t} \bigl(x_i - \bar{x}_w[t]\bigr)^2}
$$

where `n = window_size`. Matches `pandas.Series.rolling(w).std()`.


<!-- NAN_FOOTNOTE_START -->
## NaN handling

**Policy: `ignore`.** A `NaN` in any input gives a `NaN` output at that step. The step is skipped: it does not enter the window or update the internal state. See [NaN and warmup](../nan_and_warmup.md).
<!-- NAN_FOOTNOTE_END -->

## Examples

### Usage example

```{eval-rst}
.. plotly::

    import numpy as np
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from screamer import RollingStd

    np.random.seed(0)  # reproducible example data

    data = np.cumsum(np.random.normal(size=300))

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[2/3, 1/3],
        vertical_spacing=0.1
    )

    fig.add_trace(go.Scatter(y=data, mode='lines', name='Input Data'), row=1, col=1)
    fig.add_trace(go.Scatter(y=RollingStd(30)(data), mode='lines', name='Rolling Std Deviation', line=dict(color='blue')), row=2, col=1)

    fig.update_layout(
        title="Rolling Standard Deviation with Window Size 30",
        xaxis_title="Index",
        yaxis=dict(title="Input Data"),
        yaxis2=dict(title="Rolling Std Deviation"),
        margin=dict(l=20, r=20, t=80, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)        
    )

    fig.show()
```

<!-- HELP_END -->

## Implementation Details

### Algorithm

`RollingStd` keeps the window in a circular buffer with running sums of powers of `x - c`, where the origin `c` is close to the window mean, and updates the sums as each sample enters and the oldest one leaves (see [Accuracy at large levels](../conventions.md#accuracy-at-large-levels)).

### Complexity

* **Time Complexity**: `O(1)` per new element.
* **Space Complexity**: `O(window_size)`.

