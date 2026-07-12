---
name: OBV
title: On-Balance Volume (OBV)
implementation_family: rolling
topics:
- cumulative
- volume
tags:
- obv
- granville
- talib
- pair
short: 'On-Balance Volume: signed cumulative volume by close-direction (Granville,
  1963).'
inputs: 2
outputs: 1
parameters: []
nan_policy: ignore
---

# `OBV`

## Description

On-Balance Volume (Granville, 1963):

$$
\text{OBV}[t] = \text{OBV}[t-1] + \text{sign}(C - C_{t-1})\ \cdot\ V[t]
$$

with seed `OBV[0] = volume[0]` (TA-Lib's convention). **2-input, 1-output** on
`(close, volume)`. Cumulative; no window.

Bit-exact to `talib.OBV` (0.0 difference).


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
    from screamer import OBV

    np.random.seed(0)
    close = 100*np.exp(np.cumsum(np.random.normal(0, 0.01, size=300)))
    volume = np.random.uniform(1e5, 5e5, size=300)
    out = OBV()(close, volume)

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.55, 0.45], vertical_spacing=0.08)
    fig.add_trace(go.Scatter(y=close, name="close", line=dict(color="royalblue")), row=1, col=1)
    fig.add_trace(go.Scatter(y=out, name="OBV", line=dict(color="red")), row=2, col=1)
    fig.update_layout(title="On-balance volume (OBV)",
                      margin=dict(l=20, r=20, t=60, b=20),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    fig.update_yaxes(title_text="close", row=1, col=1)
    fig.update_yaxes(title_text="OBV", row=2, col=1)
    fig.show()
```

<!-- HELP_END -->
