---
name: Butter
title: Butterworth low-pass filter
implementation_family: signal
topics:
- smoothing
- filtering
tags:
- butterworth
- iir
- lowpass
- low-pass filter
- smoothing
short: General-order IIR Butterworth low-pass filter.
inputs: 1
outputs: 1
parameters:
- name: order
  type: int
  default: 2
  min: 1
  description: Filter order. A higher order gives a steeper rolloff and a longer
    delay.
- name: cutoff_freq
  type: float
  default: 0.1
  min: 0.0
  max: 1.0
  description: Cutoff frequency as a fraction of the Nyquist frequency (half the sampling
    rate), strictly between 0 and 1. 1 is the Nyquist frequency, the scipy.signal.butter Wn
    convention.
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
- ButterHighpass
- ButterBandpass
- SuperSmoother
- MovingAverage
references:
- Butterworth, S. (1930). On the theory of filter amplifiers. *Experimental Wireless and the
  Wireless Engineer*, 7, 536-541.
---

# `Butter`

## Description

`Butter` is a causal IIR Butterworth low-pass filter of configurable order. The Butterworth design has a maximally flat frequency response in the passband and attenuates components above the cutoff. Higher filter orders give a steeper rolloff.

The digital filter is the bilinear transform of the analog Butterworth prototype. It starts in the steady state for its first input, so the output starts at that value instead of rising from zero. With `start_policy="zero"` it starts from zero and matches `scipy.signal.butter(order, cutoff_freq)` followed by `scipy.signal.lfilter`.


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
    from screamer import Butter

    # Generate synthetic noisy data
    np.random.seed(0)
    data = np.cumsum(np.random.normal(0, 1, 500)) + np.sin(np.linspace(0, 10 * np.pi, 500))


    # Create subplots with original and smoothed data
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[1/3, 1/3, 1/3], vertical_spacing=0.1)

    fig.add_trace(go.Scatter(y=data, mode='lines', name='Original Data'), row=1, col=1)

    fig.add_trace(go.Scatter(
      y=Butter(order=4, cutoff_freq=0.4)(data), 
      mode='lines', name='order=4, qf=0.4', line=dict(color='purple')), row=2, col=1)   
    fig.add_trace(go.Scatter(
      y=Butter(order=4, cutoff_freq=0.1)(data), 
      mode='lines', name='order=4, qf=0.1', line=dict(color='red')), row=2, col=1)
    fig.add_trace(go.Scatter(
      y=Butter(order=4, cutoff_freq=0.05)(data), 
      mode='lines', name='order=4, qf=0.05', line=dict(color='orange')), row=2, col=1)

    fig.add_trace(go.Scatter(
      y=Butter(order=2, cutoff_freq=0.1)(data), 
      mode='lines', name='order=2, qf=0.1', line=dict(color='blue')), row=3, col=1)
    fig.add_trace(go.Scatter(
      y=Butter(order=8, cutoff_freq=0.1)(data), 
      mode='lines', name='order=8, qf=0.1', line=dict(color='green')), row=3, col=1)


    fig.update_layout(
        title="Butterworth Low-Pass Filters",
        xaxis_title="Index",
        yaxis=dict(title="Original Data"),
        yaxis2=dict(title="Cutoff freq variants"),
        yaxis3=dict(title="Order variants"),
        margin=dict(l=20, r=20, t=120, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)        
    )

    fig.show()
```

<!-- HELP_END -->

