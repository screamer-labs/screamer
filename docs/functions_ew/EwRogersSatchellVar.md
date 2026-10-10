---
name: EwRogersSatchellVar
title: EW Rogers-Satchell variance
implementation_family: ew
topics:
- volatility
tags:
- rogers-satchell
- range-based
- ohlc
- drift-robust
- var
- variance
- realized volatility
- ew
short: Exponentially weighted Rogers-Satchell drift-robust range-based variance estimator
  (OHLC).
inputs: 4
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
- EwRogersSatchellVol
- RollingRogersSatchellVar
- EwGarmanKlassVar
- EwYangZhangVar
references:
- Rogers, L. C. G. and Satchell, S. E. (1991). Estimating variance from high, low and closing
  prices. *Annals of Applied Probability*, 1(4), 504-512. doi:10.1214/aoap/1177005835
nan_policy: ignore
---

# `EwRogersSatchellVar`

## Description

`EwRogersSatchellVar` is the exponentially weighted Rogers-Satchell (1991) range-based variance estimator. It takes four inputs, `(open, high, low, close)`, and averages the per-bar variance contribution

$$
\sigma^2_\text{RS}[t] = \ln\tfrac{H}{C}\ \ln\tfrac{H}{O} + \ln\tfrac{L}{C}\ \ln\tfrac{L}{O}
$$

with an exponentially weighted mean. The smoothing factor `alpha` is set by exactly one of `com`, `span`, `halflife` or `alpha`. [`EwRogersSatchellVol`](EwRogersSatchellVol.md) returns the square root.

The Rogers-Satchell estimator does not assume zero drift, so it stays unbiased when the price has a trend within the bar. It is somewhat less efficient than Garman-Klass under zero drift (about 6 times the close-to-close estimator, against 7.4), and like Garman-Klass it does not account for overnight gaps; [`EwYangZhangVar`](EwYangZhangVar.md) adds that term.

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
    from screamer import EwRogersSatchellVar

    np.random.seed(0)
    close = 100*np.exp(np.cumsum(np.random.normal(0, 0.01, size=300)))
    open_ = np.concatenate([[close[0]], close[:-1]])
    wick = np.abs(np.random.normal(0, 0.4, size=300))
    high = np.maximum(open_, close) + wick
    low  = np.minimum(open_, close) - wick
    var = EwRogersSatchellVar(span=20)(open_, high, low, close)

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.55, 0.45], vertical_spacing=0.08)
    fig.add_trace(go.Scatter(y=close, name="close"), row=1, col=1)
    fig.add_trace(go.Scatter(y=var, name="EwRogersSatchellVar", line=dict(color="red")), row=2, col=1)
    fig.update_layout(title="EW Rogers-Satchell variance from OHLC bars (EwRogersSatchellVar)",
                      margin=dict(l=20, r=20, t=60, b=20),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    fig.update_yaxes(title_text="price", row=1, col=1)
    fig.update_yaxes(title_text="variance", row=2, col=1)
    fig.show()
```

<!-- HELP_END -->
