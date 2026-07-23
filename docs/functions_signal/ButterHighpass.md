---
name: ButterHighpass
title: Butterworth high-pass filter
implementation_family: signal
topics:
- filtering
tags:
- butterworth
- iir
- highpass
short: General-order IIR Butterworth high-pass filter (rejects low frequencies).
inputs: 1
outputs: 1
parameters:
- name: order
  type: int
  default: 2
  min: 1
  description: Filter order.
- name: cutoff_freq
  type: float
  default: 0.1
  min: 0.0
  max: 0.5
  description: Normalised cutoff.
nan_policy: ignore
---

# `ButterHighpass`

## Description

Digital high-pass Butterworth IIR filter, designed via the standard analog-prototype + `lp2hp` + bilinear pipeline.

$$
\text{ButterHighpass}(N, W_c)
$$

- `order` (int, $\ge 1$): filter order. Higher order = steeper roll-off, larger group delay.
- `cutoff_freq` (float in $(0, 1)$): normalised cutoff. `1` is the Nyquist frequency (matches `scipy.signal.butter`'s `Wn` convention).


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
    from screamer import ButterHighpass

    rng = np.random.default_rng(0)
    fs = 500.0
    N = 1000
    t = np.arange(N) / fs

    # Composite: slow drift (2 Hz) + high-frequency detail (60 Hz) + noise
    slow = 2.0 * np.sin(2 * np.pi * 2 * t)
    detail = 0.5 * np.sin(2 * np.pi * 60 * t)
    noise = 0.15 * rng.standard_normal(N)
    signal = slow + detail + noise

    # High-pass: reject below 20 Hz (normalised: 20/250=0.08)
    filtered = ButterHighpass(order=4, cutoff_freq=0.08)(signal)

    fig = go.Figure()
    fig.add_trace(go.Scatter(y=signal, mode='lines', name='Raw (2 Hz drift + 60 Hz detail)',
                             line=dict(color='lightblue'), opacity=0.7))
    fig.add_trace(go.Scatter(y=filtered, mode='lines',
                             name='ButterHighpass(order=4, cutoff_freq=0.08)',
                             line=dict(color='red')))
    fig.update_layout(
        title="Butterworth high-pass filter (cutoff: 20 Hz, rejects slow drift)",
        xaxis_title="Sample", yaxis_title="Amplitude",
        margin=dict(l=20, r=20, t=80, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.show()
```

<!-- HELP_END -->

## Implementation Details

Bit-exact match to `scipy.signal.butter(order, cutoff, btype='highpass')` + `scipy.signal.lfilter`. Verified to ~1e-12 in `tests/test_signal.py`.

Uses the same `IIRFilter` engine as the existing `Butter` low-pass class, O(order) per step.
