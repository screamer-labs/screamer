---
name: SuperSmoother
title: Ehlers SuperSmoother lowpass filter
implementation_family: signal
topics:
- smoothing
- filtering
tags:
- ehlers
- iir
- lowpass
short: Ehlers 2-pole low-lag lowpass filter.
inputs: 1
outputs: 1
parameters:
- name: period
  type: float
  default: 20.0
  description: Cutoff as a cycle length in samples. Pass exactly one of period or cutoff; the constructor has no default, and 20 is a suggested starting value.
- name: cutoff
  type: float
  default: null
  description: Cutoff as a fraction of Nyquist in (0, 1). Give this or period, not both.
nan_policy: ignore
---

# `SuperSmoother`

## Description

`SuperSmoother` is a causal 2-pole IIR lowpass filter from John Ehlers. It passes
components of the input series slower than the cutoff and attenuates faster ones,
with less lag and less ringing than a Butterworth filter of comparable rolloff.

The cutoff is set by a cycle length in samples (`period`) or, equivalently, a
fraction of the Nyquist frequency (`cutoff`), related by `cutoff = 2 / period`.
The filter coefficients follow Ehlers' critically damped design.

### Parameters

**`period`** *(float)*: The cutoff expressed as a cycle length in samples.

**`cutoff`** *(float)*: The cutoff expressed as a fraction of the Nyquist frequency, in (0, 1). Provide exactly one of `period` or `cutoff`.

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
    from screamer import SuperSmoother

    rng = np.random.default_rng(7)
    n = 400
    t = np.arange(n)
    trend = 2.0 * np.sin(2 * np.pi * t / 250) + 0.005 * t     # slow movement to keep
    x = trend + 0.4 * rng.standard_normal(n)                    # plus fast noise to remove

    smooth10 = SuperSmoother(period=10)(x)
    smooth30 = SuperSmoother(period=30)(x)

    fig = go.Figure()
    fig.add_trace(go.Scatter(y=x, name='noisy input', line=dict(color='lightgray')))
    fig.add_trace(go.Scatter(y=trend, name='underlying trend', line=dict(color='black', dash='dash')))
    fig.add_trace(go.Scatter(y=smooth10, name='SuperSmoother(period=10)', line=dict(color='steelblue')))
    fig.add_trace(go.Scatter(y=smooth30, name='SuperSmoother(period=30)', line=dict(color='crimson')))
    fig.update_layout(title='SuperSmoother: lowpass of a noisy trend at two cutoffs',
                      xaxis_title='sample', yaxis_title='value',
                      margin=dict(l=20, r=20, t=60, b=20),
                      legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1))
    fig.show()
```
