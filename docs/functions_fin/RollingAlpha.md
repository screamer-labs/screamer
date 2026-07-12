---
name: RollingAlpha
title: Rolling alpha (regression intercept)
implementation_family: fin
topics:
- regression
tags:
- alpha
- regression
- intercept
- pair
short: Rolling OLS intercept of target on regressor (companion to RollingBeta).
inputs: 2
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

# `RollingAlpha`

## Description

Companion to `RollingBeta`. The regression intercept of `y` on `x`:

$$
\alpha[t] = \overline{y}_w - \beta[t]\ \cdot\ \overline{x}_w
$$

**2-input, 1-output** on `(target, regressor)` -- same convention as `RollingBeta`.
Composes `RollingBeta` + two `RollingMean` instances. O(1) per step.


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
    from screamer import RollingAlpha

    np.random.seed(0)
    market_ret = np.random.normal(0.0004, 0.012, size=300)
    asset_ret = 0.6 * market_ret + np.random.normal(0.0003, 0.008, size=300)
    alpha = RollingAlpha(window_size=63)(asset_ret, market_ret)   # regression intercept of asset on market

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.55, 0.45], vertical_spacing=0.08)
    fig.add_trace(go.Scatter(y=asset_ret, mode="lines", name="asset returns"), row=1, col=1)
    fig.add_trace(go.Scatter(y=market_ret, mode="lines", name="market returns"), row=1, col=1)
    fig.add_trace(go.Scatter(y=alpha, mode="lines", name="alpha",
                             line=dict(color="red")), row=2, col=1)
    fig.update_layout(title="Rolling alpha of asset on market over 63 bars (RollingAlpha)",
                      margin=dict(l=20, r=20, t=60, b=20),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    fig.update_yaxes(title_text="returns", row=1, col=1)
    fig.update_yaxes(title_text="alpha", row=2, col=1)
    fig.show()
```

<!-- HELP_END -->
