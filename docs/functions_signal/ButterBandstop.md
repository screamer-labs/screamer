---
name: ButterBandstop
title: Butterworth band-stop filter
implementation_family: signal
topics:
- filtering
tags:
- butterworth
- iir
- bandstop
- notch
short: General-order IIR Butterworth band-stop (notch) filter.
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
  description: Lower stop-band edge (normalised).
- name: high_cutoff
  type: float
  default: 0.2
  min: 0.0
  max: 0.5
  description: Upper stop-band edge (normalised); must exceed low_cutoff.
nan_policy: ignore
---

# `ButterBandstop`

## Description

Digital band-stop (notch) Butterworth IIR filter -- suppresses frequencies in `[low_cutoff, high_cutoff]`. Produces a 2N-order filter from the order-N analog prototype.

$$
\text{ButterBandstop}(N, W_{\text{low}}, W_{\text{high}})
$$

- `order` (int, $\ge 1$): order of the analog prototype.
- `low_cutoff`, `high_cutoff` (floats in $(0, 1)$, with `low < high`): stop band as fractions of the Nyquist frequency.

Bit-exact match to `scipy.signal.butter(order, [low, high], btype='bandstop')` + `scipy.signal.lfilter`. Verified to ~1e-9 in `tests/test_signal.py`.


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
    from screamer import ButterBandstop

    rng = np.random.default_rng(0)
    fs = 500.0
    N = 1000
    t = np.arange(N) / fs

    # Composite: slow drift (3 Hz) + interference at 40 Hz + noise
    slow = np.sin(2 * np.pi * 3 * t)
    interference = 0.8 * np.sin(2 * np.pi * 40 * t)
    noise = 0.15 * rng.standard_normal(N)
    signal = slow + interference + noise

    # Band-stop: reject 30-55 Hz (normalised: 30/250=0.12, 55/250=0.22)
    filtered = ButterBandstop(order=4, low_cutoff=0.12, high_cutoff=0.22)(signal)

    fig = go.Figure()
    fig.add_trace(go.Scatter(y=signal, mode='lines', name='Raw (3 Hz drift + 40 Hz interference)',
                             line=dict(color='lightblue'), opacity=0.7))
    fig.add_trace(go.Scatter(y=filtered, mode='lines',
                             name='ButterBandstop(order=4, low=0.12, high=0.22)',
                             line=dict(color='red')))
    fig.update_layout(
        title="Butterworth band-stop filter (reject band: 30-55 Hz)",
        xaxis_title="Sample", yaxis_title="Amplitude",
        margin=dict(l=20, r=20, t=80, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.show()
```

<!-- HELP_END -->
