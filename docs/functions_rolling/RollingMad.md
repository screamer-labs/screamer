---
name: RollingMad
title: Rolling mean absolute deviation
implementation_family: rolling
topics:
- statistics
- outliers
tags:
- mad
- deviation
- rolling
- mean absolute deviation
- average absolute deviation
short: Trailing-window mean absolute deviation from the rolling mean.
inputs: 1
outputs: 1
parameters:
- name: window_size
  type: int
  default: 20
  min: 2
  description: Number of most recent samples whose mean and absolute deviations are taken.
- name: start_policy
  type: str
  default: strict
  enum:
  - strict
  - expanding
  - zero
  description: 'Output before the window is full: "strict" gives NaN, "expanding" computes
    from the samples seen so far, and "zero" treats the window as pre-filled with zeros.'
see_also:
- RollingMedianAD
- RollingStd
- RollingIqr
- CCI
nan_policy: ignore
---

# `RollingMad`

## Description

`RollingMad` computes the rolling mean absolute deviation from the rolling mean:

$$
\text{MAD}[t] = \frac{1}{n} \sum_{i \in \text{window}} \left| x_i - \bar{x}_t \right|
$$

It is less sensitive to outliers than [`RollingStd`](RollingStd.md) because it uses
the absolute deviation rather than the squared deviation. It is used in statistical
screening, as the denominator of a robust z-score, and as the scale in [`CCI`](CCI.md). For
the median absolute deviation from the median, use
[`RollingMedianAD`](RollingMedianAD.md).

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
    from screamer import RollingMad

    rng = np.random.default_rng(0)
    n = 400
    # Quiet then noisy regime
    x = rng.normal(0.0, 0.3, n)
    x[n//2:] = rng.normal(0.0, 1.0, n - n//2)

    mad = RollingMad(30)(x)

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.5, 0.5], vertical_spacing=0.1)
    fig.add_trace(go.Scatter(y=x, mode='lines', name='x'), row=1, col=1)
    fig.add_trace(go.Scatter(y=mad, mode='lines',
                             name='RollingMad(30)',
                             line=dict(color='red')), row=2, col=1)
    fig.update_layout(
        title="Rolling Mean Absolute Deviation Tracking a Volatility Regime Shift",
        xaxis_title="Index",
        yaxis_title="x",
        yaxis2_title="MAD",
        margin=dict(l=20, r=20, t=60, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    fig.show()
```

<!-- HELP_END -->

## Implementation Details

### Cost per step

Variance has a closed-form update (`Var = E[x²] − μ²`); the mean absolute deviation
does not. The mean shifts every step, and `|x_i − μ|` is piecewise linear in μ with a
kink at every `x_i`, so a small change in μ flips the sign of some terms but not
others. Every shift in μ therefore changes all `W` absolute deviations, not only the
entering and leaving ones.

An order-statistic tree partitioned at μ gives `O(log W)` amortized time, but it is
`O(W log W)` in the worst case and only beats the direct `O(W)` sweep on smooth inputs
with a large `W`. Pandas also uses the `O(W)` sweep.

### Algorithm

1. A circular buffer of the last `W` values with a running sum gives the mean μ in
   `O(1)`.
2. Each step sweeps the window once to accumulate `Σ |x_i − μ|`, in `O(W)`.

### Complexity

* Time complexity: `O(W)` per step.
* Space complexity: `O(window_size)`.

## Validation

Equivalent to `pandas.Series.rolling(w).apply(lambda v: np.mean(np.abs(v - v.mean())), raw=True)`
to floating-point precision. Tests check it against both pandas and a numpy reference.
