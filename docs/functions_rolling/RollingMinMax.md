---
name: RollingMinMax
title: Rolling min and max
implementation_family: rolling
topics:
- statistics
tags:
- min
- max
- rolling
- range
- channel
short: Trailing-window minimum and maximum, computed in one pass.
inputs: 1
outputs: 2
parameters:
- name: window_size
  type: int
  default: 20
  min: 2
  description: Number of most recent samples in the window.
nan_policy: ignore
output_names:
- min
- max
see_also:
- RollingMin
- RollingMax
- RollingRange
- DonchianChannels
---

# `RollingMinMax`

## Description

`RollingMinMax` returns the rolling minimum and rolling maximum of a single stream over a sliding window, both computed in one pass.

Use it when you need both extrema together (channels, breakouts, normalized range). Computing them in one pass is roughly half the work of running `RollingMin` and `RollingMax` separately.

*Equation*:

$$
\mathrm{min}_w[t] = \min_{i \in [t-w+1,\, t]} x_i
\qquad
\mathrm{max}_w[t] = \max_{i \in [t-w+1,\, t]} x_i
$$

with the window taking the most recent `window_size` samples.

*Input shape*: a single stream. Same input matrix as the 1-in/1-out functions: scalars, 1-D arrays, 2-D / N-D arrays (axis 0 is time), iterators.

*Output shape*: an extra trailing axis of size **2** is appended to the input shape. A 1-D input of shape `(T,)` returns shape `(T, 2)`; a 2-D input `(T, K)` returns `(T, K, 2)`. Index `0` along the trailing axis is the rolling minimum, index `1` is the rolling maximum.

A scalar call returns a Python `tuple` of two floats.


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
    from screamer import RollingMinMax

    np.random.seed(0)
    N = 200
    data = np.cumsum(np.random.normal(size=N))

    bands = RollingMinMax(window_size=20)(data)   # shape (N, 2)
    lo, hi = bands[:, 0], bands[:, 1]

    fig = go.Figure()
    fig.add_trace(go.Scatter(y=data, mode="lines", name="x"))
    fig.add_trace(go.Scatter(y=hi, mode="lines", name="rolling max(20)",
                             line=dict(color="green")))
    fig.add_trace(go.Scatter(y=lo, mode="lines", name="rolling min(20)",
                             line=dict(color="red")))
    fig.update_layout(
        title="Rolling minimum and maximum (window=20)",
        xaxis_title="time",
        margin=dict(l=20, r=20, t=60, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.show()
```

<!-- HELP_END -->

## Implementation Details

### Algorithm

Two monotonic deques run in parallel: one keeps the candidates for the rolling minimum (front-to-back non-decreasing), the other for the rolling maximum (front-to-back non-increasing). Each new sample either replaces some trailing candidates or extends the deque, then the front is checked for staleness against the window. Both updates are amortized O(1) per step.

### Complexity

* **Time**: `O(1)` amortized per new element.
* **Space**: `O(window_size)` worst case (each deque holds at most `window_size` entries).

### Validation

Validated against a brute-force window scan in `tests/test_one_input_multi_output.py` for windows 3, 5 and 10.
