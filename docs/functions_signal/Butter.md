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
short: General-order IIR Butterworth low-pass filter.
inputs: 1
outputs: 1
parameters:
- name: order
  type: int
  default: 2
  min: 1
  description: Filter order (1 = first-order, higher = sharper rolloff).
- name: cutoff_freq
  type: float
  default: 0.1
  min: 0.0
  max: 0.5
  description: Normalised cutoff (0 < f < 0.5, Nyquist-relative).
nan_policy: ignore
---

# `Butter`

## Description

`Butter` is a causal IIR Butterworth low-pass filter of configurable order. The Butterworth design produces a maximally flat frequency response in the passband, attenuating high-frequency components beyond a specified cutoff. Higher filter orders give a steeper rolloff.

The filter is implemented in the digital domain using a bilinear transform of the analog Butterworth poles.

### Parameters

**`order`** *(int)*: The filter order. A higher order results in a sharper frequency cutoff.

**`cutoff_freq`** *(float)*: The normalised cutoff frequency, expressed as a fraction of the Nyquist frequency (half the sampling rate). Must be in the range (0, 0.5).


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

