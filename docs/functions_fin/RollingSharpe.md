---
name: RollingSharpe
title: Rolling Sharpe ratio
implementation_family: fin
topics:
- risk
tags:
- sharpe
- ratio
- rolling
short: Annualised Sharpe ratio over a trailing window of returns.
inputs: 1
outputs: 1
parameters:
- name: window_size
  type: int
  default: 252
  min: 2
  description: Trailing-window length.
- name: periods_per_year
  type: float
  default: 1.0
  min: 1.0
  description: Annualisation factor (252 daily, 52 weekly, 12 monthly, 1 = no annualisation).
nan_policy: ignore
---

# `RollingSharpe`

## Description

Annualised Sharpe ratio over a trailing window of returns:

$$
\text{Sharpe}[t] = \sqrt{\text{ppy}}\ \cdot\ \frac{\text{RollingMean}(r)}{\text{RollingStd}(r)}
$$

Composes `RollingMean` + `RollingStd` (sample std, ddof=1 to match pandas). Returns `NaN`
where the std is zero.


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
    from screamer import RollingSharpe

    np.random.seed(0)
    ret = np.random.normal(0.0005, 0.01, size=300)
    sharpe = RollingSharpe(window_size=63, periods_per_year=252)(ret)   # annualised Sharpe over 63 bars

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.55, 0.45], vertical_spacing=0.08)
    fig.add_trace(go.Scatter(y=ret, mode="lines", name="returns"), row=1, col=1)
    fig.add_trace(go.Scatter(y=sharpe, mode="lines", name="Sharpe",
                             line=dict(color="red")), row=2, col=1)
    fig.update_layout(title="Annualised Sharpe over 63 bars (RollingSharpe)",
                      margin=dict(l=20, r=20, t=60, b=20),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    fig.update_yaxes(title_text="returns", row=1, col=1)
    fig.update_yaxes(title_text="Sharpe", row=2, col=1)
    fig.show()
```

<!-- HELP_END -->
