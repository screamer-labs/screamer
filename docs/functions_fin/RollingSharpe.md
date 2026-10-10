---
name: RollingSharpe
title: Rolling Sharpe ratio
implementation_family: fin
topics:
- risk
tags:
- sharpe
- sharpe ratio
- ratio
- risk-adjusted return
- rolling
short: Annualized Sharpe ratio over a trailing window of returns.
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
nan_policy: ignore
see_also:
- RollingSortino
- RollingInfoRatio
- RollingCalmar
- RollingMean
- RollingStd
references:
- Sharpe, W. F. (1966). Mutual fund performance. *Journal of Business*, 39(1), 119-138.
- Sharpe, W. F. (1994). The Sharpe ratio. *Journal of Portfolio Management*, 21(1), 49-58.
---

# `RollingSharpe`

## Description

`RollingSharpe` computes the annualized Sharpe ratio over a trailing window of returns:

$$
\text{Sharpe}[t] = \sqrt{\text{ppy}}\ \cdot\ \frac{\text{RollingMean}(r)}{\text{RollingStd}(r)}
$$

The standard deviation is the sample estimate (ddof=1, as in pandas). The returns are used as given, so subtract a risk-free rate first if you want excess returns. The output is `NaN` where the standard deviation is zero.


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
    from screamer import RollingSharpe

    np.random.seed(0)
    ret = np.random.normal(0.0005, 0.01, size=300)
    sharpe = RollingSharpe(window_size=63, periods_per_year=252)(ret)   # annualized Sharpe over 63 bars

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
