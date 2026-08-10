---
name: EwYangZhangVol
title: EW Yang-Zhang volatility
implementation_family: ew
topics:
- volatility
tags:
- yang-zhang
- range-based
- ohlc
- drift-robust
- gap-robust
- vol
- ew
short: Vol form (sqrt) of the EW Yang-Zhang range-based estimator.
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

# `EwYangZhangVol`

## Description

The standard-deviation form of [`EwYangZhangVar`](EwYangZhangVar.md): the square
root of the exponentially-weighted Yang-Zhang variance.

$$
\sigma_{YZ} = \sqrt{\sigma^2_{YZ}}
$$

Yang-Zhang is the most efficient classical range-based estimator. It is
drift-robust and it handles overnight gaps. This is a **4-input, 1-output**
function over `(open, high, low, close)`. See
[`EwYangZhangVar`](EwYangZhangVar.md) for the full definition and the components.

## Parameters

Specify exactly one of the following to set the smoothing factor `alpha`:

- **`com`**: Center of mass. `alpha = 1 / (1 + com)`
- **`span`**: Span. `alpha = 2 / (span + 1)`
- **`halflife`**: Half-life. `alpha = 1 - exp(-log(2) / halflife)`
- **`alpha`**: Directly sets the smoothing factor, `0 < alpha < 1`


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
    from screamer import EwYangZhangVol

    rng = np.random.default_rng(1)
    N = 500
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, N)))
    o = c * np.exp(rng.normal(0, 0.004, N))
    hi = np.maximum(o, c) * np.exp(np.abs(rng.normal(0, 0.006, N)))
    lo = np.minimum(o, c) * np.exp(-np.abs(rng.normal(0, 0.006, N)))
    vol = EwYangZhangVol(span=60)(o, hi, lo, c)

    fig = go.Figure()
    fig.add_trace(go.Scatter(y=vol, name='EwYangZhangVol(span=60)',
                             line=dict(color='crimson')))
    fig.update_layout(
        title='EW Yang-Zhang volatility from OHLC bars',
        yaxis=dict(title='per-bar volatility'),
        margin=dict(l=20, r=20, t=60, b=20),
    )
    fig.show()
```

<!-- HELP_END -->

## Reference

Yang & Zhang (2000). The square root of [`EwYangZhangVar`](EwYangZhangVar.md).
