---
name: WMA
title: Weighted Moving Average (WMA)
implementation_family: rolling
topics:
- smoothing
tags:
- wma
- linear-weighted
- moving-average
- weighted moving average
- lwma
- linear weighted moving average
short: Linearly weighted moving average.
inputs: 1
outputs: 1
parameters:
- name: window_size
  type: int
  default: 20
  min: 2
  description: Number of most recent samples in the window; the newest sample gets weight
    window_size and the oldest weight 1.
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
- EwMean
- HullMA
- TRIMA
---

# `WMA`

## Description

`WMA` computes the linearly weighted moving average: the most recent sample carries weight `w`, the one before it weight `w-1`, and so on down to weight `1` for the oldest sample in the window. The denominator is the triangular number `w*(w+1)/2`.

$$
\text{WMA}[t] = \frac{1 \cdot x[t-w+1] \;+\; 2 \cdot x[t-w+2] \;+\; \dots \;+\; w \cdot x[t]}{w \, (w+1) / 2}
$$

WMA is one of the three common moving averages, alongside `RollingMean` (SMA) and `EwMean` (EMA). `HullMA` is built from WMAs. The output before the window is full depends on `start_policy`, described under Warmup behavior below.

## Implementation Details

### Why O(1) per step?

WMA admits a closed-form rolling recurrence. Let `W[t] = 1·x[t-w+1] + ... + w·x[t]` be the linear-weighted sum, and `S[t-1]` the simple rolling sum of the *previous* window (i.e. `x[t-1] + x[t-2] + ... + x[t-w]`). Then

$$
W[t] - W[t-1] \;=\; w \cdot x[t] \;-\; S[t-1]
$$

(every old weight drops by 1, contributing `-S[t-1]`, and the new sample enters with weight `w`). `WMA` therefore keeps a rolling sum for `S` and a single number for `W`.

### Complexity

* Time complexity: `O(1)` per step.
* Space complexity: `O(window_size)` for the rolling-sum buffer.

### Warmup behavior

While the window is filling (n samples seen, `n < window_size`):

| Policy | Output during warmup |
|---|---|
| `"strict"` (default) | `NaN` |
| `"expanding"` | partial-window WMA: weights `1, 2, ..., n`, divisor `n(n+1)/2` |
| `"zero"` | weights `1, 2, ..., w` with implicit zeros for missing past, full divisor `w(w+1)/2` (output is dampened during warmup) |

The warmup numerators agree exactly when the window first fills, so the output is continuous at the switch to the recurrence.


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
    from screamer import WMA

    rng = np.random.default_rng(0)
    N = 300
    data = np.cumsum(rng.standard_normal(N))

    fig = go.Figure()
    fig.add_trace(go.Scatter(y=data, mode='lines', name='Input', line=dict(color='steelblue', width=1)))
    fig.add_trace(go.Scatter(y=WMA(window_size=20)(data), mode='lines', name='WMA(window_size=20)', line=dict(color='crimson', width=2)))
    fig.update_layout(
        title="WMA smoother over a random walk",
        xaxis_title="Index", yaxis_title="Value",
        margin=dict(l=20, r=20, t=80, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.show()
```

<!-- HELP_END -->

## Compatibility

Equivalent to `pandas.Series.rolling(w).apply(lambda v: np.dot(v, np.arange(1, w+1)) / (w*(w+1)/2))` and to TA-Lib's `WMA`. Validated in `tests/test_wma.py` against three brute-force per-window references (one per policy) and against pandas to floating-point precision.
