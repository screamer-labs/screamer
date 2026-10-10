---
name: RollingLinearRegression
title: Rolling linear regression
implementation_family: fin
topics:
- trend
- regression
tags:
- regression
- ols
- linear-regression
- least squares
- r-squared
- pair
short: Rolling OLS fit of y on x, returning slope, intercept, R-squared and the standard error
  of the fit.
inputs: 2
outputs: 4
parameters:
- name: window_size
  type: int
  default: 20
  min: 3
  description: Number of most recent (y, x) pairs in the regression window. Must be at least
    3, because the standard error divides by n - 2.
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
output_names:
- slope
- intercept
- r_squared
- stderr
see_also:
- RollingBeta
- RollingAlpha
- RollingPoly1
- RollingTSF
- KalmanRegression
---

# `RollingLinearRegression`

## Description

`RollingLinearRegression` fits `y ≈ slope · x + intercept + ε` by ordinary least squares over a trailing window. It takes two inputs, `(y, x)`, and returns four columns per step:

| Output | Formula |
|---|---|
| `slope` | $(n S_{xy} - S_x S_y) / (n S_{xx} - S_x^2)$ |
| `intercept` | $(S_y - \text{slope} \cdot S_x) / n$ |
| `r_squared` | $(n S_{xy} - S_x S_y)^2 / [(n S_{xx} - S_x^2)(n S_{yy} - S_y^2)]$ |
| `stderr` | $\sqrt{\text{SSE} / (n-2)}$, the residual standard error |

Here `n` is the number of samples in the window and `S` denotes window sums. `stderr` is the standard error of the *estimate*, not the standard error of the *slope coefficient*; multiply it by `1/sqrt(Σ(x - mean_x)²)` to get the latter. With the default `start_policy="strict"` the first valid output is at index `window_size - 1`. All four outputs are `NaN` when `x` is constant in the window, and `r_squared` is `NaN` when `y` is constant.


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
    from screamer import RollingLinearRegression

    np.random.seed(0)
    x = np.random.normal(0.0004, 0.012, size=300)
    y = 0.8 * x + np.random.normal(0.0003, 0.007, size=300)
    out = RollingLinearRegression(window_size=63)(y, x)   # columns: slope, intercept, r_squared, stderr
    slope = out[:, 0]

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.55, 0.45], vertical_spacing=0.08)
    fig.add_trace(go.Scatter(y=y, mode="lines", name="y returns"), row=1, col=1)
    fig.add_trace(go.Scatter(y=x, mode="lines", name="x returns"), row=1, col=1)
    fig.add_trace(go.Scatter(y=slope, mode="lines", name="slope",
                             line=dict(color="red")), row=2, col=1)
    fig.update_layout(title="Rolling OLS slope of y on x over 63 bars (RollingLinearRegression)",
                      margin=dict(l=20, r=20, t=60, b=20),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    fig.update_yaxes(title_text="returns", row=1, col=1)
    fig.update_yaxes(title_text="slope", row=2, col=1)
    fig.show()
```

<!-- HELP_END -->
