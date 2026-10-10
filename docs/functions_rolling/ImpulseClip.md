---
name: ImpulseClip
title: Impulse clip
implementation_family: rolling
topics:
- outliers
tags:
- outlier
- despike
- impulse
- glitch
- robust
- rolling
- spike removal
- glitch filter
- outlier filter
short: Causal impulse remover, detects spikes on the trend-free first difference.
inputs: 1
outputs: 1
parameters:
- name: window_size
  type: int
  default: 20
  min: 1
  description: Number of most recent first differences (and values) used for the robust scale
    and the replacement median.
- name: n_sigma
  type: float
  default: 4.0
  min: 0
  description: Detection threshold in robust standard deviations of the differences (1.4826
    * MAD). Larger values flag fewer samples; 4.0 is typical.
- name: output
  type: str
  default: cleaned
  enum:
  - cleaned
  - flag
  - nan
  description: 'What to return: "cleaned" gives the signal with outliers replaced by the median,
    "flag" gives 1.0 at outliers and 0.0 elsewhere, and "nan" gives the input with outliers
    replaced by NaN.'
- name: start_policy
  type: str
  default: strict
  description: 'Output before the window is full: "strict" gives NaN, "expanding" computes
    from the samples seen so far, and "zero" treats the window as pre-filled with zeros.'
  enum:
  - strict
  - expanding
  - zero
see_also:
- Hampel
- RollingMedianAD
- Diff
- RollingSigmaClip
nan_policy: ignore
---

# `ImpulseClip`

## Description

`ImpulseClip` removes impulse spikes (glitches) from a signal that may be strongly
non-stationary. A spike is a large, isolated jump: hard to separate from the signal's
own swing in the raw level, but obvious in the sample-to-sample change. Detection
therefore runs on the first difference, whose scale is free of trend:

$$ d_t = x_t - x_{t-1}, \qquad
   \text{outlier if } |d_t| > \texttt{n\_sigma} \times 1.4826 \times \text{MAD}(d)_t, $$

where `MAD(d)` is the rolling median absolute deviation of the differences. A flagged
sample is replaced by the trailing median of the values. The filter is causal and has
zero latency.

Detecting without looking ahead has two consequences. An impulse is a `+/-` doublet in
the difference (a jump onto the spike and a jump back), so both the spike and its
return sample are flagged, and two consecutive samples are replaced (the second nudged
to the median). A genuine level shift keeps its body, but its onset sample is clipped.
Avoiding both would need a one-sample lookahead, which `ImpulseClip` does not use.
For stationary data or outliers that span several samples, use [`Hampel`](Hampel.md).

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
    from screamer import ImpulseClip

    rng = np.random.default_rng(0)
    N = 300
    data = np.cumsum(rng.standard_normal(N))
    data[[100, 150, 200]] += 15

    cleaned = ImpulseClip(window_size=21, n_sigma=4.0)(data)

    fig = go.Figure()
    fig.add_trace(go.Scatter(y=data, mode='lines', name='Raw (with spikes)', line=dict(color='steelblue')))
    fig.add_trace(go.Scatter(y=cleaned, mode='lines', name='ImpulseClip(window_size=21, n_sigma=4.0)', line=dict(color='crimson')))
    fig.update_layout(
        title='ImpulseClip: causal impulse remover on a random walk',
        xaxis_title='Index', yaxis_title='Value',
        margin=dict(l=20, r=20, t=60, b=20),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
    )
    fig.show()
```

<!-- HELP_END -->

## Implementation Details

`O(W)` per step. Two trailing windows are kept: one of the values (for the replacement
median) and one of the first differences (for the robust scale). Each step takes the
median absolute deviation of the differences with `std::nth_element` to set the
threshold, tests the current jump, and on a flag replaces the sample with the median of
the values, which is also written back into the window.
