---
name: RoofingFilter
title: Ehlers roofing filter (bandpass)
implementation_family: signal
topics:
- filtering
tags:
- ehlers
- iir
- bandpass
short: Ehlers bandpass, a highpass then a SuperSmoother.
inputs: 1
outputs: 1
parameters:
- name: hp_period
  type: float
  default: 48.0
  description: Highpass cutoff as a cycle length in samples. Pass exactly one of hp_period or hp_cutoff; the constructor has no default, and 48 is a suggested starting value.
- name: lp_period
  type: float
  default: 10.0
  description: Lowpass cutoff as a cycle length in samples. Pass exactly one of lp_period or lp_cutoff; the constructor has no default, and 10 is a suggested starting value.
- name: hp_cutoff
  type: float
  default: null
  description: Highpass cutoff as a fraction of Nyquist in (0, 1). Give this or hp_period.
- name: lp_cutoff
  type: float
  default: null
  description: Lowpass cutoff as a fraction of Nyquist in (0, 1). Give this or lp_period.
nan_policy: ignore
---

# `RoofingFilter`

## Description

`RoofingFilter` is a causal bandpass from John Ehlers. It applies a 2-pole
highpass at `hp_period` to remove the trend, then a `SuperSmoother` at
`lp_period` to remove fast noise, leaving the band of cycles between the two
cutoffs. It is the preprocessing step Ehlers applies before measuring a cycle,
because a raw series carries a trend and high-frequency noise that distort the
measurement.

Each cutoff is a cycle length in samples (`hp_period`, `lp_period`) or a fraction
of the Nyquist frequency (`hp_cutoff`, `lp_cutoff`), related by `cutoff = 2 / period`.

### Parameters

**`hp_period`** / **`hp_cutoff`** *(float)*: The highpass cutoff. Provide exactly one.

**`lp_period`** / **`lp_cutoff`** *(float)*: The lowpass cutoff. Provide exactly one.

<!-- NAN_FOOTNOTE_START -->
## NaN handling

**Policy: `ignore`.** A `NaN` in the input at index `t` causes the function to skip that step: output at `t` is `NaN` and internal state is unchanged. Subsequent finite samples are processed as if step `t` had not occurred.
<!-- NAN_FOOTNOTE_END -->

## Examples

### Usage plot

```{eval-rst}
.. plotly::
    :include-source: True

    import numpy as np
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from screamer import RoofingFilter

    rng = np.random.default_rng(20)
    n = 400
    t = np.arange(n)
    tone = np.sin(2 * np.pi * t / 20)                           # cycle inside the 10-48 band
    x = tone + 0.01 * t + 0.3 * rng.standard_normal(n)         # plus a trend and fast noise

    band = RoofingFilter(hp_period=48, lp_period=10)(x)

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.06)
    fig.add_trace(go.Scatter(y=x, name='tone + trend + noise', line=dict(color='gray')), row=1, col=1)
    fig.add_trace(go.Scatter(y=tone, name='20-sample tone', line=dict(color='black', dash='dash')), row=2, col=1)
    fig.add_trace(go.Scatter(y=band, name='RoofingFilter(hp_period=48, lp_period=10)',
                             line=dict(color='steelblue')), row=2, col=1)
    fig.update_layout(title='RoofingFilter: trend and noise removed, the cycle band kept',
                      yaxis=dict(title='input'), yaxis2=dict(title='bandpass'),
                      xaxis2=dict(title='sample'),
                      margin=dict(l=20, r=20, t=60, b=20),
                      legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1))
    fig.show()
```
