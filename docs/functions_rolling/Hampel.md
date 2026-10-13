---
name: Hampel
title: Hampel filter
implementation_family: rolling
topics:
- filtering
- outliers
tags:
- outlier
- despike
- hampel
- robust
- rolling
- outlier filter
- hampel identifier
- spike removal
short: Robust Hampel despiker, replace samples far from the window median (in MAD units).
inputs: 1
outputs: 1
parameters:
- name: window_size
  type: int
  default: 20
  min: 1
  description: Number of most recent samples used for the median and the MAD.
- name: n_sigma
  type: float
  default: 3.0
  min: 0
  description: Detection threshold in robust standard deviations (1.4826 * MAD). Larger values
    flag fewer samples; 3.0 is typical. inf flags no sample, so the output is the input.
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
- ImpulseClip
- RollingMedian
- RollingMedianAD
- RollingSigmaClip
references:
- Hampel, F. R. (1974). The influence curve and its role in robust estimation. *Journal of
  the American Statistical Association*, 69(346), 383-393. doi:10.1080/01621459.1974.10482962
nan_policy: ignore
---

# `Hampel`

## Description

`Hampel` is the Hampel filter (also called the Hampel identifier), a robust despiker,
in its causal trailing-window form. Over the trailing window it computes the median
`m` and the median absolute deviation `MAD`, and flags a sample as an outlier when

$$ \bigl| x_t - m_t \bigr| > \texttt{n\_sigma} \times 1.4826 \times \text{MAD}_t $$

The factor `1.4826` makes the MAD a consistent estimate of the standard deviation for
Gaussian data. Because the filter uses the median and MAD rather than the mean and
standard deviation, a few spikes cannot drag the center or inflate the scale. A
flagged sample is replaced by the window median, and the replacement, not the raw
outlier, is fed back into the window, so a burst of spikes does not pollute later
scale estimates.

When the window is constant the MAD is zero, so no sample is flagged (there is no
scale to measure against). For strongly non-stationary signals see
[`ImpulseClip`](ImpulseClip.md), which detects on the trend-free first difference.

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
    from screamer import Hampel

    rng = np.random.default_rng(0)
    N = 300
    data = np.cumsum(rng.standard_normal(N))
    data[[100, 150, 200]] += 15

    cleaned = Hampel(window_size=21, n_sigma=3.0)(data)

    fig = go.Figure()
    fig.add_trace(go.Scatter(y=data, mode='lines', name='Raw (with spikes)', line=dict(color='steelblue')))
    fig.add_trace(go.Scatter(y=cleaned, mode='lines', name='Hampel(window_size=21, n_sigma=3.0)', line=dict(color='crimson')))
    fig.update_layout(
        title='Hampel filter: robust despiker on a random walk',
        xaxis_title='Index', yaxis_title='Value',
        margin=dict(l=20, r=20, t=60, b=20),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
    )
    fig.show()
```

<!-- HELP_END -->

## Implementation Details

`O(W)` per step: each step copies the trailing window, computes the median and the
median absolute deviation with `std::nth_element`, and applies the threshold. Detected
outliers are written back into the window as the median so they do not bias later
medians.
