---
name: MaxDrawdown
title: Maximum drawdown
implementation_family: fin
topics:
- cumulative
- risk
tags:
- drawdown
- max-drawdown
short: Worst drawdown experienced so far (since reset).
inputs: 1
outputs: 1
parameters: []
nan_policy: ignore
---

# `MaxDrawdown`

## Description

The worst (most negative) drawdown ever observed since the start (or last `reset()`):

$$
\text{MaxDrawdown}[t] = \min_{k \le t}\ \text{Drawdown}[k]
$$

Monotonically non-increasing in time. Composes `Drawdown` + `CumMin`.

## Notes

- For the trailing-window equivalent see `RollingMaxDrawdown`.


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
    from screamer import MaxDrawdown

    np.random.seed(0)
    price = 100 * np.exp(np.cumsum(np.random.normal(0.0005, 0.02, size=300)))
    mdd = MaxDrawdown()(price)              # worst drawdown seen so far, only gets deeper

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.55, 0.45], vertical_spacing=0.08)
    fig.add_trace(go.Scatter(y=price, mode="lines", name="price"), row=1, col=1)
    fig.add_trace(go.Scatter(y=mdd, mode="lines", name="max drawdown",
                             line=dict(color="red"), fill="tozeroy"), row=2, col=1)
    fig.update_layout(title="Worst drawdown so far (MaxDrawdown)",
                      margin=dict(l=20, r=20, t=60, b=20),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    fig.update_yaxes(title_text="price", row=1, col=1)
    fig.update_yaxes(title_text="max drawdown", row=2, col=1)
    fig.show()
```

<!-- HELP_END -->
