---
name: RollingLinearRegression
title: Rolling linear regression (2->4 OLS fit)
implementation_family: fin
topics:
- trend
- regression
tags:
- regression
- ols
- linear-regression
- pair
short: Full OLS fit returning (slope, intercept, r_squared, stderr).
inputs: 2
outputs: 4
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

# `RollingLinearRegression`

## Description

Rolling OLS fit of `y ≈ slope · x + intercept + ε`, returning four outputs per step:

| Output | Formula |
|---|---|
| `slope` | $(n S_{xy} - S_x S_y) / (n S_{xx} - S_x^2)$ |
| `intercept` | $(S_y - \text{slope} \cdot S_x) / n$ |
| `r_squared` | $(n S_{xy} - S_x S_y)^2 / [(n S_{xx} - S_x^2)(n S_{yy} - S_y^2)]$ |
| `stderr` | $\sqrt{\text{SSE} / (n-2)}$ - RMSE of residuals |

**Note on `stderr`**: this is the standard error of the *estimate* (RMSE of the fit), not
the standard error of the *slope coefficient*. Multiply by `1/sqrt(Σ(x - mean_x)²)` to get
slope confidence.

**2-input, 4-output** (`FunctorBase<_, 2, 4>`). Inputs `(y, x)`; outputs
`(slope, intercept, r_squared, stderr)`. First valid output at sample index
`window_size - 1`. First 2→4 consumer of the Plan E `N→M` dispatcher.


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
