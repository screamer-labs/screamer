---
name: Drawdown
title: Drawdown
implementation_family: fin
topics:
- cumulative
- risk
tags:
- drawdown
- underwater
- peak to trough
short: Running drawdown from the cumulative peak.
inputs: 1
outputs: 1
parameters: []
nan_policy: ignore
see_also:
- MaxDrawdown
- RollingMaxDrawdown
- CumMax
- RollingCalmar
---

# `Drawdown`

## Description

Running drawdown from the cumulative-since-inception peak:

$$
\text{Drawdown}[t] = \frac{\text{price}[t]}{\text{CumMax}(\text{price})[t]} - 1
$$

A flat or new-high series gives `0`. A 30% loss from the prior peak gives `-0.30`.
There is no warmup. The result matches `price / price.cummax() - 1` in pandas.


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
    from screamer import Drawdown

    np.random.seed(0)
    price = 100 * np.exp(np.cumsum(np.random.normal(0.0005, 0.02, size=300)))
    dd = Drawdown()(price)                  # 0 at a new peak, negative in a drawdown

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.6, 0.4], vertical_spacing=0.08)
    fig.add_trace(go.Scatter(y=price, mode="lines", name="price"), row=1, col=1)
    fig.add_trace(go.Scatter(y=dd, mode="lines", name="drawdown",
                             line=dict(color="red"), fill="tozeroy"), row=2, col=1)
    fig.update_layout(title="Running drawdown from the peak (Drawdown)",
                      margin=dict(l=20, r=20, t=60, b=20),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    fig.update_yaxes(title_text="price", row=1, col=1)
    fig.update_yaxes(title_text="drawdown", row=2, col=1)
    fig.show()
```

<!-- HELP_END -->
