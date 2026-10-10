---
name: RollingCalmar
title: Rolling Calmar ratio
implementation_family: fin
topics:
- risk
tags:
- calmar
- calmar ratio
- ratio
- rolling
- drawdown
- risk-adjusted return
short: Annualized mean return divided by the worst drawdown in a trailing window.
inputs: 1
outputs: 1
parameters:
- name: window_size
  type: int
  default: 252
  min: 2
  description: Number of most recent returns used for the mean return and the drawdown.
- name: periods_per_year
  type: float
  default: 1.0
  min: 1.0
  description: 'Annualization factor that multiplies the mean return: 252 for daily, 52 for
    weekly, 12 for monthly data, 1 for no annualization.'
nan_policy: ignore
see_also:
- RollingMaxDrawdown
- RollingSharpe
- RollingSortino
- MaxDrawdown
---

# `RollingCalmar`

## Description

`RollingCalmar` computes the Calmar ratio over a trailing window: the annualized mean return divided by the worst drawdown in the window.

$$
\text{Calmar}[t] = \frac{\text{ppy}\ \cdot\ \text{mean}(r)}{\big|\,\text{RollingMaxDrawdown}(\text{implied price})\,\big|}
$$

The input is a series of simple returns. The drawdown is measured on the implied price path, the cumulative product of `1 + r` starting from 1.0. The output is `NaN` until the window is full and when the window contains no drawdown. For a price series, convert it to returns with [`Return`](Return.md) first.


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
    from screamer import RollingCalmar

    np.random.seed(0)
    ret = np.random.normal(0.0005, 0.01, size=300)
    calmar = RollingCalmar(window_size=63, periods_per_year=252)(ret)   # annualized return over worst rolling drawdown

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
