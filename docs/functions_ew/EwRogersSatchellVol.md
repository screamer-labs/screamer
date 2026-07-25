---
name: EwRogersSatchellVol
title: EW Rogers-Satchell volatility
implementation_family: ew
topics:
- volatility
tags:
- rogers-satchell
- range-based
- ohlc
- drift-robust
- vol
- ew
short: Vol form of the Rogers-Satchell drift-robust range-based estimator.
inputs: 4
outputs: 1
parameters:
- name: com
  type: float
  default: null
  description: Center of mass.
- name: span
  type: float
  default: 20.0
  description: Span.
- name: halflife
  type: float
  default: null
  description: Halflife.
- name: alpha
  type: float
  default: null
  description: Smoothing parameter directly.
nan_policy: ignore
---

# `EwRogersSatchellVol`

## Description

`EwRogersSatchellVol` returns the square root of the Rogers-Satchell (1991) range-based
variance estimate: `sqrt(EwRogersSatchellVar)`. The per-bar variance contribution is:

$$
\sigma^2_\text{RS}[t] = \ln\tfrac{H}{C}\ \ln\tfrac{H}{O} + \ln\tfrac{L}{C}\ \ln\tfrac{L}{O}
$$

This expression is averaged with an exponentially-weighted mean to form the estimator, and the
square root is taken to produce a volatility in the same units as the log-price process.

**4-input, 1-output** on `(open, high, low, close)`. Slightly less efficient (~6x vs
close-to-close `RollingVar`) than Garman-Klass, but **drift-robust**: it does not assume
zero drift, so it remains consistent when the underlying process has a non-zero mean.


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
    from screamer import EwRogersSatchellVol

    np.random.seed(0)
    close = 100*np.exp(np.cumsum(np.random.normal(0, 0.01, size=300)))
    open_ = np.concatenate([[close[0]], close[:-1]])
    wick = np.abs(np.random.normal(0, 0.4, size=300))
    high = np.maximum(open_, close) + wick
    low  = np.minimum(open_, close) - wick
    vol = EwRogersSatchellVol(span=20)(open_, high, low, close)

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.55, 0.45], vertical_spacing=0.08)
    fig.add_trace(go.Scatter(y=close, name="close"), row=1, col=1)
    fig.add_trace(go.Scatter(y=vol, name="EwRogersSatchellVol", line=dict(color="red")), row=2, col=1)
    fig.update_layout(title="EW Rogers-Satchell volatility from OHLC bars (EwRogersSatchellVol)",
                      margin=dict(l=20, r=20, t=60, b=20),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    fig.update_yaxes(title_text="price", row=1, col=1)
    fig.update_yaxes(title_text="volatility", row=2, col=1)
    fig.show()
```

<!-- HELP_END -->
