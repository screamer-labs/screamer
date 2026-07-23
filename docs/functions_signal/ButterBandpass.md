---
name: ButterBandpass
title: Butterworth band-pass filter
implementation_family: signal
topics:
- filtering
tags:
- butterworth
- iir
- bandpass
short: General-order IIR Butterworth band-pass filter.
inputs: 1
outputs: 1
parameters:
- name: order
  type: int
  default: 2
  min: 1
  description: Filter order.
- name: low_cutoff
  type: float
  default: 0.05
  min: 0.0
  max: 0.5
  description: Lower cutoff frequency (normalised).
- name: high_cutoff
  type: float
  default: 0.2
  min: 0.0
  max: 0.5
  description: Upper cutoff frequency (normalised); must exceed low_cutoff.
nan_policy: ignore
---

# `ButterBandpass`

## Description

Digital band-pass Butterworth IIR filter. Produces a 2N-order filter from the order-N analog prototype via the `lp2bp` transformation.

$$
\text{ButterBandpass}(N, W_{\text{low}}, W_{\text{high}})
$$

- `order` (int, $\ge 1$): order of the analog prototype. Effective digital order is `2 * order`.
- `low_cutoff`, `high_cutoff` (floats in $(0, 1)$, with `low < high`): pass band as fractions of the Nyquist frequency.

Bit-exact match to `scipy.signal.butter(order, [low, high], btype='bandpass')` + `scipy.signal.lfilter`. Verified to ~1e-9 in `tests/test_signal.py`.


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
    from screamer import ButterBandpass

    rng = np.random.default_rng(0)
    fs = 500.0
    N = 1000
    t = np.arange(N) / fs

    # Composite: low-frequency component (5 Hz) + high-frequency component (80 Hz) + noise
    low_freq = np.sin(2 * np.pi * 5 * t)
    high_freq = 0.5 * np.sin(2 * np.pi * 80 * t)
    noise = 0.2 * rng.standard_normal(N)
    signal = low_freq + high_freq + noise

    # Band-pass: keep 20-60 Hz (normalised: 20/250=0.08, 60/250=0.24)
    filtered = ButterBandpass(order=4, low_cutoff=0.08, high_cutoff=0.24)(signal)

    fig = go.Figure()
    fig.add_trace(go.Scatter(y=signal, mode='lines', name='Raw (5 Hz + 80 Hz + noise)',
                             line=dict(color='lightblue'), opacity=0.7))
    fig.add_trace(go.Scatter(y=filtered, mode='lines',
                             name='ButterBandpass(order=4, low=0.08, high=0.24)',
                             line=dict(color='red')))
    fig.update_layout(
        title="Butterworth band-pass filter (pass band: 20-60 Hz)",
        xaxis_title="Sample", yaxis_title="Amplitude",
        margin=dict(l=20, r=20, t=80, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.show()
```

<!-- HELP_END -->
