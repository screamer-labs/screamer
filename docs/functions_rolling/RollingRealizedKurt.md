---
name: RollingRealizedKurt
title: Rolling realized kurtosis
implementation_family: rolling
topics:
- statistics
tags:
- realized kurtosis
- kurtosis
- higher moments
- realized moments
- intraday returns
- tail risk
- amaya
- christoffersen
short: Realized kurtosis of a window of returns, N times the sum of fourth powers of the returns
  over the squared realized variance.
inputs: 1
outputs: 1
parameters:
- name: window_size
  type: int
  default: 20
  min: 2
  description: Number of most recent returns in the window.
- name: start_policy
  type: str
  default: strict
  enum:
  - strict
  - expanding
  - zero
  description: 'Output before the window is full: "strict" gives NaN, "expanding" uses the
    returns seen so far, and "zero" treats the window as pre-filled with zeros.'
nan_policy: ignore
see_also:
- RollingRealizedSkew
- RollingKurt
- RollingJumpShare
references:
- 'Amaya, D., Christoffersen, P., Jacobs, K. and Vasquez, A. (2015). Does realized skewness predict
  the cross-section of equity returns? *Journal of Financial Economics* 118(1), 135-167.'
---

# `RollingRealizedKurt`

## Description

`RollingRealizedKurt` is the realized kurtosis of a window of `N` returns
(Amaya, Christoffersen, Jacobs and Vasquez 2015):

$$
RV = \sum_{i=1}^{N} r_i^2, \qquad
RKurt = \frac{N \sum_{i=1}^{N} r_i^4}{RV^{2}}
$$

The input is returns, typically intraday returns, with the window spanning the
period the measure describes. `N` is `window_size` once the window is full.

`RKurt` is the kurtosis, not the excess kurtosis: normal returns give about 3.
The moments are taken about zero and have no small-sample correction. This is
where the measure differs from [`RollingKurt`](RollingKurt.md), which reports
the bias-corrected excess kurtosis about the window mean. The value does not
change when the returns are scaled.

A window whose returns are all zero has `RV = 0` and gives `NaN`.

<!-- NAN_FOOTNOTE_START -->
## NaN handling

**Policy: `ignore`.** A `NaN` in any input gives a `NaN` output at that step. The step is skipped: it does not enter the window or update the internal state. See [NaN and warmup](../nan_and_warmup.md).
<!-- NAN_FOOTNOTE_END -->

## Examples

### Usage plot

```{eval-rst}
.. plotly::
    :include-source: True

    import numpy as np
    import plotly.graph_objects as go
    from screamer import RollingRealizedKurt

    rng = np.random.default_rng(6)
    r = rng.standard_normal(78 * 40) * 0.001             # 40 days of 5-minute returns
    r[[78 * 10 + 30, 78 * 25 + 5]] = [0.02, -0.025]     # two jumps
    kurt = RollingRealizedKurt(78)(r)

    fig = go.Figure()
    fig.add_trace(go.Scatter(y=kurt, name='realized kurtosis, 1-day window', line=dict(color='navy')))
    fig.add_hline(y=3, line=dict(color='gray', dash='dot'))
    fig.update_layout(title='RollingRealizedKurt: a jump in the window lifts it far above 3',
                      xaxis_title='5-minute step', yaxis_title='kurtosis',
                      margin=dict(l=20, r=20, t=60, b=20),
                      legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1))
    fig.show()
```

<!-- HELP_END -->
