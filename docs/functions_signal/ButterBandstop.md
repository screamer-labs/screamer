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
- band-stop filter
- notch filter
short: General-order IIR Butterworth band-stop (notch) filter.
inputs: 1
outputs: 1
parameters:
- name: order
  type: int
  default: 2
  min: 1
  description: Order N of the analog prototype. The digital filter has order 2N.
- name: low_cutoff
  type: float
  default: 0.05
  min: 0.0
  max: 1.0
  description: Lower edge of the stop band, as a fraction of the Nyquist frequency. Must satisfy
    0 < low_cutoff < high_cutoff < 1.
- name: high_cutoff
  type: float
  default: 0.2
  min: 0.0
  max: 1.0
  description: Upper edge of the stop band, as a fraction of the Nyquist frequency. Must satisfy
    0 < low_cutoff < high_cutoff < 1.
nan_policy: ignore
see_also:
- ButterBandpass
- Butter
- ButterHighpass
references:
- Butterworth, S. (1930). On the theory of filter amplifiers. *Experimental Wireless and the
  Wireless Engineer*, 7, 536-541.
---

# `ButterBandstop`

## Description

Digital band-stop (notch) Butterworth IIR filter. It suppresses frequencies in `[low_cutoff, high_cutoff]` and produces a 2N-order filter from the order-N analog prototype.

$$
\text{ButterBandstop}(N, W_{\text{low}}, W_{\text{high}})
$$

The cutoffs are fractions of the Nyquist frequency. The output matches `scipy.signal.butter(order, [low, high], btype='bandstop')` followed by `scipy.signal.lfilter` to within about 1e-9.


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
