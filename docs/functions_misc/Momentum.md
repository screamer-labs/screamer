---
name: Momentum
title: Momentum
implementation_family: misc
topics:
- returns
- momentum
tags:
- momentum
- mom
- talib
- price change
short: x[t] - x[t-k], TA-Lib's MOM. Mathematically identical to Diff(k).
inputs: 1
outputs: 1
parameters:
- name: window_size
  type: int
  default: 10
  min: 1
  description: Lookback k, in samples. TA-Lib also defaults to 10.
- name: start_policy
  type: str
  default: strict
  enum:
  - strict
  - expanding
  - zero
  description: 'Output before the window is full: "strict" gives NaN, "expanding" computes
    from the samples seen so far, and "zero" treats the window as pre-filled with zeros.'
nan_policy: propagate
see_also:
- Diff
- ROC
- ROCP
- Return
---

# `Momentum`

## Description

`Momentum(k)` returns the *raw price displacement* over `k` steps:

$$
\text{Momentum}[t] = x[t] - x[t-k]
$$

This is the same computation as [`Diff(k)`](Diff.md), under the name TA-Lib uses (`MOM`). The two give identical output, including the `start_policy` behavior, and both match `talib.MOM` exactly. The default lookback of 10 follows TA-Lib.


<!-- NAN_FOOTNOTE_START -->
## NaN handling

**Policy: `propagate`.** A `NaN` input is stored in the lookback like any other value. The output is `NaN` wherever the formula references it and recovers once it leaves the lookback. See [NaN and warmup](../nan_and_warmup.md).
<!-- NAN_FOOTNOTE_END -->

## Examples

### Usage example

```{eval-rst}
.. plotly::
    :include-source: True

    import numpy as np
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from screamer import Momentum

    np.random.seed(0)
    x = np.cumsum(np.random.normal(size=200))
    mom = Momentum(window_size=20)(x)

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.55, 0.45], vertical_spacing=0.08)
    fig.add_trace(go.Scatter(y=x, name="price"), row=1, col=1)
    fig.add_trace(go.Scatter(y=mom, name="Momentum", line=dict(color="red")), row=2, col=1)
    fig.update_layout(title="Change over the last 20 steps (Momentum)",
                      margin=dict(l=20, r=20, t=60, b=20),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    fig.update_yaxes(title_text="price", row=1, col=1)
    fig.update_yaxes(title_text="momentum", row=2, col=1)
    fig.show()
```

<!-- HELP_END -->
