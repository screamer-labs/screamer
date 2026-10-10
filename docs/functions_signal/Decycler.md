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
short: Ehlers trend estimate with short cycles removed.
inputs: 1
outputs: 1
parameters:
- name: period
  type: float
  default: 60.0
  description: Cycles at or below this length in samples are removed. Pass exactly one of period or cutoff; the constructor has no default, and 60 is a suggested starting value.
- name: cutoff
  type: float
  default: null
  description: Cutoff as a fraction of Nyquist in (0, 1). Give this or period, not both.
nan_policy: ignore
---

# `Decycler`

## Description

`Decycler` is a causal filter from John Ehlers that estimates the trend of a
series by removing cycles at or below the cutoff. It is the input minus its
1-pole highpass component, so slow movements pass through and fast cycles are
suppressed.

The cutoff is set by a cycle length in samples (`period`) or a fraction of the
Nyquist frequency (`cutoff`), related by `cutoff = 2 / period`.

### Parameters

**`period`** *(float)*: The shortest cycle length in samples that is removed.

**`cutoff`** *(float)*: The cutoff as a fraction of the Nyquist frequency, in (0, 1). Provide exactly one of `period` or `cutoff`.

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
