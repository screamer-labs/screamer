---
name: RollingMad
title: Rolling mean absolute deviation
implementation_family: rolling
topics:
- statistics
tags:
- mad
- deviation
- rolling
short: Trailing-window mean absolute deviation from the rolling mean.
inputs: 1
outputs: 1
parameters:
- name: window_size
  type: int
  default: 20
  min: 2
  description: Trailing-window length.
- name: start_policy
  type: str
  default: strict
  enum:
  - strict
  - expanding
  - zero
  description: Warmup behaviour.
nan_policy: ignore
---

# `RollingMad`

## Description

`RollingMad` computes the rolling **mean absolute deviation**:

$$
\text{MAD}[t] = \frac{1}{n} \sum_{i \in \text{window}} \left| x_i - \bar{x}_t \right|
$$

A robust scale measure: less sensitive to outliers than `RollingStd` because it uses the absolute deviation rather than the squared deviation. Common in statistical screening and as a denominator in robust z-scores.

*Parameters*:

- `window_size` (int, positive).
- `start_policy` (str, optional): `"strict"` (default), `"expanding"`, or `"zero"`. Same semantics as `RollingMean`.

*NaN handling*: NaN values should be preprocessed.


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

### Why O(W) and not O(1)?

Unlike variance (`Var = E[x²] − μ²`), MAD has **no closed-form recurrence**. The mean shifts every step, and `|x_i − μ|` is piecewise-linear in μ with kinks at every `x_i` -- so a small change in μ flips signs on some terms but not others. Every shift in μ propagates into all `W` absolute deviations, not just the entering / leaving one.

A geometric route gives `O(log W)` *amortised* (using an order-statistic tree partitioned at μ), but it is `O(W log W)` worst-case and only beats the naive O(W) on smooth inputs with large `W`. Pandas uses the naive O(W) version too.

### Algorithm

1. Maintain a circular buffer of the last `W` values plus a running sum (the same rolling-sum recurrence `RollingMean` uses internally) -- gives the mean μ in `O(1)`.
2. On each step, loop over the window once to accumulate `Σ |x_i − μ|` -- `O(W)`.

### Complexity

* Time complexity: `O(W)` per step.
* Space complexity: `O(window_size)`.

## Reference

Equivalent to `pandas.Series.rolling(w).apply(lambda v: np.mean(np.abs(v - v.mean())), raw=True)` to floating-point precision. Validated in tests against both pandas and a manual numpy reference.
