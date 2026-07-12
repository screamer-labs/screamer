---
name: RollingSortino
title: Rolling Sortino ratio
implementation_family: fin
topics:
- risk
tags:
- sortino
- ratio
- rolling
- downside
short: 'Annualised Sortino ratio: Sharpe with downside-only deviation.'
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
- name: target
  type: float
  default: 0.0
  description: Minimum acceptable return (only deviations below this contribute to
    the denominator).
nan_policy: ignore
---

# `RollingSortino`

## Description

Annualised Sortino ratio:

$$
\text{Sortino}[t] = \sqrt{\text{ppy}}\ \cdot\ \frac{\text{mean}(r) - \text{target}}{\sqrt{\text{mean}(\min(r - \text{target},\ 0)^2)}}
$$

Same as Sharpe but the denominator is the *downside* deviation -- only bars below `target`
contribute, so upside variability is not penalised.

## Implementation

`O(window_size)` per step. The downside-RMS denominator does not have a closed-form
O(1) update.


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
    from screamer import RollingSortino

    np.random.seed(0)
    ret = np.random.normal(0.0005, 0.01, size=300)
    sortino = RollingSortino(window_size=63, periods_per_year=252)(ret)   # Sharpe using only downside deviation

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.55, 0.45], vertical_spacing=0.08)
    fig.add_trace(go.Scatter(y=ret, mode="lines", name="returns"), row=1, col=1)
    fig.add_trace(go.Scatter(y=sortino, mode="lines", name="Sortino",
                             line=dict(color="red")), row=2, col=1)
    fig.update_layout(title="Annualised Sortino over 63 bars (RollingSortino)",
                      margin=dict(l=20, r=20, t=60, b=20),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    fig.update_yaxes(title_text="returns", row=1, col=1)
    fig.update_yaxes(title_text="Sortino", row=2, col=1)
    fig.show()
```

<!-- HELP_END -->
