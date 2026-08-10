---
name: EwYangZhangVar
title: EW Yang-Zhang variance
implementation_family: ew
topics:
- volatility
tags:
- yang-zhang
- range-based
- ohlc
- drift-robust
- gap-robust
- var
- ew
short: Var form of the Yang-Zhang range-based estimator, exponentially weighted.
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

# `EwYangZhangVar`

## Description

The exponentially-weighted Yang-Zhang (2000) volatility estimator. It is the EW
analog of [`RollingYangZhangVar`](../functions_fin/RollingYangZhangVar.md) and
completes the EW range-based volatility family
([`EwParkinsonVar`](EwParkinsonVar.md), [`EwGarmanKlassVar`](EwGarmanKlassVar.md),
[`EwRogersSatchellVar`](EwRogersSatchellVar.md)).

Yang-Zhang is the most efficient classical range-based estimator. It is
drift-robust (through the Rogers-Satchell term) and it handles overnight gaps
(through the overnight term). It combines three parts:

$$
\sigma^2_{YZ} = \sigma^2_o + k\,\sigma^2_c + (1-k)\,\sigma^2_{RS}
$$

- $\sigma^2_o$ is the EW variance of the overnight log returns $\ln(O_t / C_{t-1})$.
- $\sigma^2_c$ is the EW variance of the open-to-close log returns $\ln(C_t / O_t)$.
- $\sigma^2_{RS}$ is the EW mean of the per-bar Rogers-Satchell term.
- $k = 0.34 / (1.34 + (n_\text{eff}+1)/(n_\text{eff}-1))$.

The rolling estimator fixes $k$ from the window size. This estimator uses the
effective sample size $n_\text{eff}$ of the overnight leg, so $k$ adapts as the
weighting fills. This is a **4-input, 1-output** function over `(open, high,
low, close)`.

## Parameters

Specify exactly one of the following to set the smoothing factor `alpha`:

- **`com`**: Center of mass. `alpha = 1 / (1 + com)`
- **`span`**: Span. `alpha = 2 / (span + 1)`
- **`halflife`**: Half-life. `alpha = 1 - exp(-log(2) / halflife)`
- **`alpha`**: Directly sets the smoothing factor, `0 < alpha < 1`

The first overnight return needs a prior close, so the output is `NaN` until at
least two overnight returns exist.


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

    rng = np.random.default_rng(0)
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

Yang & Zhang (2000). Composes two [`EwVar`](EwVar.md) (overnight and
open-to-close) and one [`EwMean`](EwMean.md) (Rogers-Satchell per bar); `O(1)`
per step. Use [`EwYangZhangVol`](EwYangZhangVol.md) for the standard-deviation
form.
