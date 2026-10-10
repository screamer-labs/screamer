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
- triple exponential moving average
- moving average
short: 'Mulloy''s Triple EMA: 3*EMA - 3*EMA(EMA) + EMA(EMA(EMA)).'
inputs: 1
outputs: 1
parameters:
- name: com
  type: float
  default: null
  description: Center of mass c, giving alpha = 1 / (1 + c). Must be positive. Give exactly
    one of com, span, halflife and alpha.
- name: span
  type: float
  default: 20.0
  description: Span s in samples, giving alpha = 2 / (s + 1). Must be greater than 1. Give
    exactly one of com, span, halflife and alpha. The default shown is an example value; the
    constructor has no default.
- name: halflife
  type: float
  default: null
  description: Half-life h in samples, giving alpha = 1 - exp(-ln(2) / h). Must be positive.
    Give exactly one of com, span, halflife and alpha.
- name: alpha
  type: float
  default: null
  description: Smoothing factor, with 0 < alpha < 1; a larger alpha gives recent samples more
    weight. Give exactly one of com, span, halflife and alpha.
nan_policy: ignore
see_also:
- DEMA
- EwMean
- TRIX
- HullMA
- KAMA
references:
- Mulloy, P. G. (1994). Smoothing data with less lag. *Technical Analysis of Stocks & Commodities*,
  12(2).
---

# `TEMA`

## Description

`TEMA` (triple exponential moving average, Patrick Mulloy, 1994) extends the `DEMA` construction by one more level:

$$
\text{TEMA}[t] = 3 \cdot \text{EMA}(x)[t] - 3 \cdot \text{EMA}(\text{EMA}(x))[t] + \text{EMA}(\text{EMA}(\text{EMA}(x)))[t]
$$

The three-term combination reduces lag further: `TEMA` typically tracks the input faster than `DEMA`, and much faster than a plain EMA of the same span, at the cost of less smoothing. The decay parameter (exactly one of `com`, `span`, `halflife` or `alpha`, as in `EwMean`) is shared by all three EMAs.

## Implementation Details

### Algorithm

`TEMA` chains three `EwMean` stages. It has no warmup: each `EwMean` returns a valid value from t=0, so `TEMA[0] = 3*x[0] - 3*x[0] + x[0] = x[0]`.

### Complexity

* Time complexity: `O(1)` per step.
* Space complexity: `O(1)`.


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

## Compatibility

Equivalent to TA-Lib's `TEMA`. Validated in `tests/test_moving_averages.py` against the explicit composition for four span values.
