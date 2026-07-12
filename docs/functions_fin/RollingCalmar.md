---
name: RollingCalmar
title: Rolling Calmar ratio
implementation_family: fin
topics:
- risk
tags:
- calmar
- ratio
- rolling
- drawdown
short: Annualised return divided by the worst rolling drawdown.
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

# `RollingCalmar`

## Description

Calmar ratio: annualised return divided by the worst rolling drawdown:

$$
\text{Calmar}[t] = \frac{\text{ppy}\ \cdot\ \text{mean}(r)}{\big|\,\text{RollingMaxDrawdown}(\text{implied price})\,\big|}
$$

Takes a *returns* series; internally reconstructs the implied price path as a cumulative
product `price *= (1 + r)` starting from 1.0, so the drawdown calculation is well-defined.
Returns `NaN` when the path is monotonic up (no drawdown in window).

If you already have a price series, compose by hand:


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
    from screamer import RollingCalmar

    np.random.seed(0)
    ret = np.random.normal(0.0005, 0.01, size=300)
    calmar = RollingCalmar(window_size=63, periods_per_year=252)(ret)   # annualised return over worst rolling drawdown

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.55, 0.45], vertical_spacing=0.08)
    fig.add_trace(go.Scatter(y=ret, mode="lines", name="returns"), row=1, col=1)
    fig.add_trace(go.Scatter(y=calmar, mode="lines", name="Calmar",
                             line=dict(color="red")), row=2, col=1)
    fig.update_layout(title="Annualised Calmar over 63 bars (RollingCalmar)",
                      margin=dict(l=20, r=20, t=60, b=20),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    fig.update_yaxes(title_text="returns", row=1, col=1)
    fig.update_yaxes(title_text="Calmar", row=2, col=1)
    fig.show()
```

<!-- HELP_END -->
