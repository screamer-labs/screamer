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
- low-pass filter
- smoothing
short: Ehlers 2-pole low-lag lowpass filter.
inputs: 1
outputs: 1
parameters:
- name: period
  type: float
  default: 20.0
  description: Cutoff as a cycle length in samples; cycles shorter than this are attenuated.
    Give exactly one of period and cutoff. The default shown is a suggested starting value;
    the constructor has no default.
- name: cutoff
  type: float
  default: null
  description: Cutoff as a fraction of the Nyquist frequency, in (0, 1), equal to 2 / period.
    Give exactly one of period and cutoff.
- name: start_policy
  type: str
  default: steady
  enum:
  - steady
  - zero
  description: 'Initial filter state. "steady" sets the state, at the first non-NaN input x0, to the
    steady state for a constant input x0, so the output starts at x0. "zero" starts from
    an all-zero state, as scipy.signal.lfilter does.'
nan_policy: ignore
see_also:
- Butter
- Decycler
- RoofingFilter
- EwMean
references:
- 'Ehlers, J. F. (2013). *Cycle Analytics for Traders: Advanced Technical Trading Concepts*.
  Wiley.'
---

# `SuperSmoother`

## Description

`SuperSmoother` is a causal 2-pole IIR lowpass filter from John Ehlers. It passes
components of the input series slower than the cutoff and attenuates faster ones,
with less lag and less ringing than a Butterworth filter of comparable rolloff.

The cutoff is set by a cycle length in samples (`period`) or, equivalently, a
fraction of the Nyquist frequency (`cutoff`), related by `cutoff = 2 / period`.
The filter coefficients follow Ehlers' critically damped design.

The filter starts in the steady state for its first input, so the output starts
at that value instead of rising from zero. `start_policy="zero"` starts from
zero instead.

<!-- NAN_FOOTNOTE_START -->
## NaN handling

**Policy: `ignore`.** A `NaN` in any input gives a `NaN` output at that step. The step is skipped: it does not enter the window or update the internal state. See [NaN and warmup](../nan_and_warmup.md).
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
