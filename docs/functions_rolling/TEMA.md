---
name: TEMA
title: Triple Exponential MA (TEMA)
implementation_family: rolling
topics:
- smoothing
- trend
tags:
- tema
- ema
- mulloy
- moving-average
short: 'Mulloy''s Triple EMA: 3*EMA - 3*EMA(EMA) + EMA(EMA(EMA)).'
inputs: 1
outputs: 1
parameters:
- name: com
  type: float
  default: null
  description: Center of mass. Exclusive with span/halflife/alpha.
- name: span
  type: float
  default: 20.0
  description: Span (default smoothing parameter).
- name: halflife
  type: float
  default: null
  description: Halflife.
- name: alpha
  type: float
  default: null
  description: Smoothing parameter.
nan_policy: ignore
---

# `TEMA`

## Description

`TEMA` (Triple Exponential Moving Average, Patrick Mulloy 1994) extends the `DEMA` construction by one more level:

$$
\text{TEMA}[t] = 3 \cdot \text{EMA}(x)[t] - 3 \cdot \text{EMA}(\text{EMA}(x))[t] + \text{EMA}(\text{EMA}(\text{EMA}(x)))[t]
$$

The three-term combination further reduces lag: `TEMA` typically tracks faster than `DEMA` and much faster than a plain EMA of the same span, in exchange for slightly less smoothing.

## Parameters

Same `com / span / halflife / alpha` mutex as `EwMean` -- specify exactly one. The same value is used for all three internal EMAs.

## Implementation Details

### Algorithm

Pure composition of three chained `EwMean` instances. No explicit warmup: each `EwMean` returns a valid value from t=0, so `TEMA[0] = 3*x[0] - 3*x[0] + x[0] = x[0]`.

### Complexity

* Time complexity: `O(1)` per step.
* Space complexity: `O(1)`.


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
    from screamer import TEMA

    rng = np.random.default_rng(0)
    N = 300
    data = np.cumsum(rng.standard_normal(N))

    fig = go.Figure()
    fig.add_trace(go.Scatter(y=data, mode='lines', name='Input', line=dict(color='steelblue', width=1)))
    fig.add_trace(go.Scatter(y=TEMA(span=20)(data), mode='lines', name='TEMA(span=20)', line=dict(color='crimson', width=2)))
    fig.update_layout(
        title="TEMA smoother over a random walk",
        xaxis_title="Index", yaxis_title="Value",
        margin=dict(l=20, r=20, t=80, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.show()
```

<!-- HELP_END -->

## Reference

Equivalent to TA-Lib's `TEMA`. Validated in `tests/test_moving_averages.py` against the explicit composition for four span values.
