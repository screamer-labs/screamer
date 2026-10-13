---
name: RollingHitRate
title: Rolling hit rate
implementation_family: fin
topics:
- risk
tags:
- hit-rate
- hit rate
- win-rate
- win rate
- rolling
short: Fraction of strictly positive values in a trailing window.
inputs: 1
outputs: 1
parameters:
- name: window_size
  type: int
  default: 252
  min: 1
  description: Number of most recent values counted.
nan_policy: ignore
see_also:
- RollingMean
- RollingSharpe
- RollingOmega
- GreaterThan
---

# `RollingHitRate`

## Description

`RollingHitRate` is the fraction of strictly positive values in the trailing window:

$$
\text{HitRate}[t] = \frac{1}{w}\ \text{count}(r_i > 0,\ i \in \text{window})
$$

The output lies in `[0, 1]`. Zero counts as a miss.


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
    from screamer import RollingHitRate

    np.random.seed(0)
    ret = np.random.normal(0.0005, 0.01, size=300)
    hit = RollingHitRate(window_size=63)(ret)   # fraction of positive returns in the last 63 bars

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.55, 0.45], vertical_spacing=0.08)
    fig.add_trace(go.Scatter(y=ret, mode="lines", name="returns"), row=1, col=1)
    fig.add_trace(go.Scatter(y=hit, mode="lines", name="hit rate",
                             line=dict(color="red")), row=2, col=1)
    fig.update_layout(title="Fraction of positive returns over 63 bars (RollingHitRate)",
                      margin=dict(l=20, r=20, t=60, b=20),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    fig.update_yaxes(title_text="returns", row=1, col=1)
    fig.update_yaxes(title_text="hit rate", row=2, col=1)
    fig.show()
```

<!-- HELP_END -->
