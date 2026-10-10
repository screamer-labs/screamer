---
name: Lag
title: Lag (delay)
implementation_family: misc
topics:
- returns
tags:
- lag
- delay
- shift
- lagged
short: Output is the input delayed by k samples.
inputs: 1
outputs: 1
parameters:
- name: window_size
  type: int
  default: 1
  min: 1
  description: Number of samples k to delay the input by.
- name: start_policy
  type: str
  default: strict
  enum:
  - strict
  - expanding
  - zero
  description: 'Output before the window is full: "strict" gives NaN, "expanding" computes
    from the samples seen so far, and "zero" treats the window as pre-filled with zeros.'
nan_policy: propagate
see_also:
- Diff
- Momentum
- Hold
- Delay
---

# `Lag`

## Description

The `Lag` function returns the values from a sequence offset by a specified number of steps backward. It shifts each element in the input by `window_size` positions.

*Equation*:

$$
y[i] = x[i - k]
$$

where `k` is `window_size`. The first `k` outputs have no `x[t - k]`; `start_policy` decides what they are. Under `"strict"` they are `NaN`, under `"expanding"` they repeat the first sample, and under `"zero"` they are 0.


<!-- NAN_FOOTNOTE_START -->
## NaN handling

**Policy: `propagate`.** A `NaN` input is stored in the lookback like any other value. The output is `NaN` wherever the formula references it and recovers once it leaves the lookback. See [NaN and warmup](../nan_and_warmup.md).
<!-- NAN_FOOTNOTE_END -->

## Examples

### Usage example

```{eval-rst}
.. plotly::
    :include-source: True

    import numpy as np
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from screamer import Lag

    np.random.seed(0)  # reproducible example data

    data = np.random.normal(size=50)
    lagged_data = Lag(5)(data)

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[1/2, 1/2], vertical_spacing=0.1)

    fig.add_trace(go.Scatter(y=data, mode='lines+markers', name='Original Data'), row=1, col=1)
    fig.add_trace(go.Scatter(y=lagged_data, mode='lines+markers', name='Lagged by 5', line=dict(color='red')), row=2, col=1)

    fig.update_layout(
        title="Lagged Data (5-step Delay)",
        xaxis_title="Index",
        yaxis_title="Original Data",
        yaxis2_title="Lagged Data",
        margin=dict(l=20, r=20, t=60, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    fig.show()
```

<!-- HELP_END -->

## Implementation Details

`Lag` keeps the last `window_size` samples in a circular buffer and returns the oldest, at O(1) cost per step.
