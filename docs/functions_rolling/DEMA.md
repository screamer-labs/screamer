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
- double exponential moving average
short: 'Mulloy''s Double EMA: 2*EMA - EMA(EMA).'
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
see_also:
- EwMean
- TEMA
- HullMA
- KAMA
- TRIX
references:
- Mulloy, P. G. (1994). Smoothing data with faster moving averages. *Technical Analysis of
  Stocks & Commodities*, 12(1), 11-19.
nan_policy: ignore
---

# `DEMA`

## Description

`DEMA` (Double Exponential Moving Average, Mulloy 1994) is a low-lag smoother:

$$
\text{DEMA}[t] = 2 \cdot \text{EMA}(x)[t] - \text{EMA}(\text{EMA}(x))[t]
$$

Subtracting the EMA of the EMA from twice the EMA removes most of the lag of a single
EMA. The result follows the input more closely than a plain EMA of the same span while
still smoothing high-frequency noise. Both EMAs are [`EwMean`](../functions_ew/EwMean.md)
with the same decay, set by one of `com`, `span`, `halflife` or `alpha`.

There is no warmup. `EwMean` returns `x[0]` at the first sample, so `DEMA[0] = x[0]`.
Each step costs `O(1)` time and memory.

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

## Validation

Matches the explicit `2*EwMean - EwMean(EwMean)` composition for four span values in
`tests/test_moving_averages.py`. TA-Lib's `DEMA` seeds its EMA differently and differs
during the early samples; see [conventions](../conventions.md).
