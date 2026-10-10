---
name: RollingSortino
title: Rolling Sortino ratio
implementation_family: fin
topics:
- risk
tags:
- sortino
- sortino ratio
- ratio
- rolling
- downside
- downside risk
short: 'Annualized Sortino ratio: the Sharpe ratio with downside deviation in the denominator.'
inputs: 1
outputs: 1
parameters:
- name: window_size
  type: int
  default: 252
  min: 2
  description: Number of most recent returns in the window.
- name: periods_per_year
  type: float
  default: 1.0
  min: 1.0
  description: 'Annualization factor: the ratio is multiplied by sqrt(periods_per_year). Use
    252 for daily, 52 for weekly, 12 for monthly data, and 1 for no annualization.'
- name: target
  type: float
  default: 0.0
  description: Minimum acceptable return. It is subtracted from the mean, and only returns
    below it contribute to the denominator.
nan_policy: ignore
see_also:
- RollingSharpe
- RollingDownsideDeviation
- RollingCalmar
- RollingOmega
references:
- Sortino, F. A. and Price, L. N. (1994). Performance measurement in a downside risk framework.
  *Journal of Investing*, 3(3), 59-64.
---

# `RollingSortino`

## Description

`RollingSortino` computes the annualized Sortino ratio over a trailing window of returns:

$$
\text{Sortino}[t] = \sqrt{\text{ppy}}\ \cdot\ \frac{\text{mean}(r) - \text{target}}{\sqrt{\text{mean}(\min(r - \text{target},\ 0)^2)}}
$$

It differs from the Sharpe ratio in the denominator, which is the *downside* deviation: only returns below `target` contribute. Each step costs O(window_size), because the downside deviation has no O(1) update.


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
