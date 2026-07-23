---
name: RollingMedianAD
title: Rolling median absolute deviation
implementation_family: rolling
topics:
- statistics
- outliers
tags:
- robust
- scale
- dispersion
- rolling
short: Rolling median absolute deviation, median(|x - median|), a robust scale estimate.
inputs: 1
outputs: 1
parameters:
- name: window_size
  type: int
  default: 20
  min: 1
  description: Trailing-window length.
- name: start_policy
  type: str
  default: strict
  description: Warmup handling before the window is full (strict, expanding, or zero).
nan_policy: ignore
---

# `RollingMedianAD`

## Description

`RollingMedianAD` computes the rolling **median absolute deviation** (MAD): for the
trailing window it takes the median of the absolute deviations from the window
median,

$$ \text{MAD}_t = \text{median}_{i \in W_t}\; \bigl| x_i - \text{median}(W_t) \bigr|. $$

Unlike [`RollingMad`](RollingMad.md), which is the *mean* absolute deviation, this is
a **robust** scale estimate: a few extreme samples cannot inflate it. It is the scale
primitive behind the [`Hampel`](Hampel.md) and [`ImpulseClip`](ImpulseClip.md)
despikers. The raw MAD is returned; multiply by `1.4826` for a Gaussian-consistent
standard-deviation estimate.

*Parameters*:
- **`window_size`**: *(int)* Trailing-window length. Must be positive.
- **`start_policy`**: How the initial phase is handled before `window_size` samples
  are available:
  - `"strict"`: `NaN` until the window is full.
  - `"expanding"`: compute over the samples seen so far, from the first sample.
  - `"zero"`: treat missing samples as zeros (a full window of zeros to start).

<!-- NAN_FOOTNOTE_START -->
## NaN handling

**Policy: `ignore`.** A `NaN` in any input at index `t` causes the function to skip
that step: output at `t` is `NaN` and internal state is unchanged. Subsequent finite
samples are processed as if step `t` had not occurred.
<!-- NAN_FOOTNOTE_END -->

## Examples

### Usage example

```{eval-rst}
.. plotly::
    :include-source: True

    import numpy as np
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from screamer import RollingMedianAD

    rng = np.random.default_rng(0)
    N = 300
    data = np.cumsum(rng.standard_normal(N))

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.5, 0.5],
                        vertical_spacing=0.08)
    fig.add_trace(go.Scatter(y=data, mode='lines', name='Input', line=dict(color='steelblue', width=1)),
                  row=1, col=1)
    fig.add_trace(go.Scatter(y=RollingMedianAD(window_size=30)(data), mode='lines',
                             name='RollingMedianAD(window_size=30)',
                             line=dict(color='crimson', width=2)), row=2, col=1)
    fig.update_layout(
        title="Rolling median absolute deviation over a random walk",
        yaxis=dict(title='Input'), yaxis2=dict(title='MAD'),
        margin=dict(l=20, r=20, t=60, b=20),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
    )
    fig.show()
```

<!-- HELP_END -->

## Implementation Details

O(W) per step: the window median shifts every step, so the absolute deviations
cannot be updated incrementally. Each step copies the active window, finds the median
with `std::nth_element`, then finds the median of the absolute deviations the same
way. Strictly causal.
