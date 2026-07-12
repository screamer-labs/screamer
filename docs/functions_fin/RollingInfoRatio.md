---
name: RollingInfoRatio
title: Rolling information ratio
implementation_family: fin
topics:
- risk
tags:
- info-ratio
- active-return
- rolling
- pair
short: 'Annualised information ratio: Sharpe of active returns against a benchmark.'
inputs: 2
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

# `RollingInfoRatio`

## Description

Information ratio against a benchmark:

$$
\text{IR}[t] = \sqrt{\text{ppy}}\ \cdot\ \frac{\text{mean}(r - b)}{\text{std}(r - b)}
$$

**2-input, 1-output** on `(returns, benchmark)`. Effectively `RollingSharpe` applied to the
active-return series `r - b`.


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
    from screamer import RollingInfoRatio

    np.random.seed(0)
    bench_ret = np.random.normal(0.0004, 0.012, size=300)
    asset_ret = bench_ret + np.random.normal(0.0003, 0.006, size=300)
    info = RollingInfoRatio(window_size=63)(asset_ret, bench_ret)   # Sharpe of active returns vs benchmark

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.55, 0.45], vertical_spacing=0.08)
    fig.add_trace(go.Scatter(y=asset_ret, mode="lines", name="asset returns"), row=1, col=1)
    fig.add_trace(go.Scatter(y=bench_ret, mode="lines", name="benchmark returns"), row=1, col=1)
    fig.add_trace(go.Scatter(y=info, mode="lines", name="info ratio",
                             line=dict(color="red")), row=2, col=1)
    fig.update_layout(title="Rolling information ratio over 63 bars (RollingInfoRatio)",
                      margin=dict(l=20, r=20, t=60, b=20),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    fig.update_yaxes(title_text="returns", row=1, col=1)
    fig.update_yaxes(title_text="info ratio", row=2, col=1)
    fig.show()
```

<!-- HELP_END -->
