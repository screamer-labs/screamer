---
name: RollingLinearRegression
title: Rolling linear regression (2->4 OLS fit)
implementation_family: fin
topics:
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

<!-- HELP_END -->
