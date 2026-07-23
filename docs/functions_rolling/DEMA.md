---
name: DEMA
title: Double Exponential MA (DEMA)
implementation_family: rolling
topics:
- smoothing
- trend
tags:
- dema
- ema
- mulloy
- moving-average
short: 'Mulloy''s Double EMA: 2*EMA - EMA(EMA).'
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

# `DEMA`

## Description

`DEMA` (Double Exponential Moving Average, Patrick Mulloy 1994) is a low-lag smoother defined as:

$$
\text{DEMA}[t] = 2 \cdot \text{EMA}(x)[t] - \text{EMA}(\text{EMA}(x))[t]
$$

The construction subtracts the *lag* of a single EMA from twice that EMA. The result tracks the input more closely than a plain EMA of the same span, while still smoothing high-frequency noise.

## Parameters

Same `com / span / halflife / alpha` mutex as `EwMean` -- specify exactly one. The same value is used for both internal EMAs.

## Implementation Details

### Algorithm

`DEMA` is a pure composition of two chained `EwMean` instances. The class holds them as members and combines their outputs as `2*e1 - e2`. There is no warmup: each `EwMean` returns a valid value from t=0 (`sum_x / sum_w` is `x[0]/1 = x[0]` after one append), so `DEMA[0] = 2*x[0] - x[0] = x[0]`.

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
    from screamer import DEMA

    rng = np.random.default_rng(0)
    N = 300
    data = np.cumsum(rng.standard_normal(N))

    fig = go.Figure()
    fig.add_trace(go.Scatter(y=data, mode='lines', name='Input', line=dict(color='steelblue', width=1)))
    fig.add_trace(go.Scatter(y=DEMA(span=20)(data), mode='lines', name='DEMA(span=20)', line=dict(color='crimson', width=2)))
    fig.update_layout(
        title="DEMA smoother over a random walk",
        xaxis_title="Index", yaxis_title="Value",
        margin=dict(l=20, r=20, t=80, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.show()
```

<!-- HELP_END -->

## Reference

Equivalent to TA-Lib's `DEMA`. Validated in `tests/test_moving_averages.py` against the explicit `2*EwMean - EwMean(EwMean)` composition for four span values.
