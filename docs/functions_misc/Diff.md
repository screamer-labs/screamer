---
name: Diff
title: Discrete difference
implementation_family: misc
topics:
- returns
tags:
- diff
- difference
- first difference
- delta
- change
short: x[t] - x[t-k] (first difference at lag k).
inputs: 1
outputs: 1
parameters:
- name: window_size
  type: int
  default: 1
  min: 1
  description: Lag k of the difference, in samples; 1 gives the difference between consecutive
    samples.
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
- Diff2
- Momentum
- Lag
- Return
- LogReturn
---

# `Diff`

## Description

The `Diff` function calculates the difference between each element in a sequence and the element `window_size` positions before it.

*Equation*:

$$
y[i] = x[i] - x[i - k]
$$

where `k` is `window_size`. The first `k` outputs have no `x[t - k]`; `start_policy` decides what they are. Under `"expanding"` the missing values are taken to equal the first sample, so those outputs are `x[t] - x[0]`, and under `"zero"` they are taken to be 0, so the outputs equal `x[t]`.


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
    from screamer import Diff

    np.random.seed(0)  # reproducible example data

    data = np.cumsum(np.random.normal(size=50) + 0.5)
    diff_data = Diff(3)(data)

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[1/2, 1/2], vertical_spacing=0.1)

    fig.add_trace(go.Scatter(y=data, mode='lines+markers', name='Original Data'), row=1, col=1)
    fig.add_trace(go.Scatter(y=diff_data, mode='lines+markers', name='Diff (3-step)', line=dict(color='green')), row=2, col=1)

    fig.update_layout(
        title="Difference Computation (3-step Delay)",
        xaxis_title="Index",
        yaxis_title="Original Data",
        yaxis2_title="Difference",
        margin=dict(l=20, r=20, t=60, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    fig.show()
```

<!-- HELP_END -->

## Implementation Details

`Diff` keeps the last `window_size` samples in a circular buffer and returns the newest minus the oldest, at O(1) cost per step.