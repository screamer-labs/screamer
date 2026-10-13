---
name: Decycler
title: Ehlers Decycler trend filter
implementation_family: signal
topics:
- smoothing
- filtering
tags:
- ehlers
- iir
- trend
- trend filter
- high-pass
short: Ehlers trend estimate with short cycles removed.
inputs: 1
outputs: 1
parameters:
- name: period
  type: float
  default: 60.0
  description: Cutoff as a cycle length in samples, the half-power point; cycles well below this length are removed.
    Give exactly one of period and cutoff. The default shown is a suggested starting value;
    the constructor has no default.
- name: cutoff
  type: float
  default: null
  description: Cutoff as a fraction of the Nyquist frequency, in (0, 1), equal to 2 / period.
    Give exactly one of period and cutoff.
nan_policy: ignore
see_also:
- SuperSmoother
- RoofingFilter
- ButterHighpass
- Detrend
---

# `Decycler`

## Description

`Decycler` is a causal filter from John Ehlers that estimates the trend of a
series by removing its short cycles. It is the input minus its 1-pole highpass
component, so slow movements pass through and fast cycles are suppressed. The
cutoff is the half-power point: a cycle of exactly `period` samples keeps
about 71% of its amplitude (-3 dB), and cycles well below the cutoff are
removed.

The cutoff is set by a cycle length in samples (`period`) or a fraction of the
Nyquist frequency (`cutoff`), related by `cutoff = 2 / period`.

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
    from screamer import Decycler

    rng = np.random.default_rng(11)
    n = 500
    t = np.arange(n)
    trend = 3.0 * np.sin(2 * np.pi * t / 400) + 0.004 * t      # slow trend
    cycle = 0.8 * np.sin(2 * np.pi * t / 20)                    # 20-sample cycle
    x = trend + cycle + 0.2 * rng.standard_normal(n)

    decycled = Decycler(period=60)(x)                           # remove cycles of 60 samples and shorter

    fig = go.Figure()
    fig.add_trace(go.Scatter(y=x, name='trend + 20-sample cycle + noise', line=dict(color='lightgray')))
    fig.add_trace(go.Scatter(y=trend, name='underlying trend', line=dict(color='black', dash='dash')))
    fig.add_trace(go.Scatter(y=decycled, name='Decycler(period=60)', line=dict(color='steelblue')))
    fig.update_layout(title='Decycler: trend estimate with the 20-sample cycle removed',
                      xaxis_title='sample', yaxis_title='value',
                      margin=dict(l=20, r=20, t=60, b=20),
                      legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1))
    fig.show()
```
