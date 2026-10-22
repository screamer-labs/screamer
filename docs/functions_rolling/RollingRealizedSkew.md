---
name: RollingRealizedSkew
title: Rolling realized skewness
implementation_family: rolling
topics:
- statistics
tags:
- realized skewness
- skewness
- higher moments
- realized moments
- intraday returns
- amaya
- christoffersen
short: Realized skewness of a window of returns, sqrt(N) times the sum of cubed returns over the
  realized variance to the power 3/2.
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
- RollingRealizedKurt
- RollingSkew
- RollingBipowerVar
references:
- 'Amaya, D., Christoffersen, P., Jacobs, K. and Vasquez, A. (2015). Does realized skewness predict
  the cross-section of equity returns? *Journal of Financial Economics* 118(1), 135-167.'
---

# `RollingRealizedSkew`

## Description

`RollingRealizedSkew` is the realized skewness of a window of `N` returns
(Amaya, Christoffersen, Jacobs and Vasquez 2015):

$$
RV = \sum_{i=1}^{N} r_i^2, \qquad
RSkew = \frac{\sqrt{N} \sum_{i=1}^{N} r_i^3}{RV^{3/2}}
$$

The input is returns, typically intraday returns, with the window spanning the
period the measure describes (for example the 78 five-minute returns of a
trading day, or a week of them). `N` is `window_size` once the window is full.

The moments are taken about zero and have no small-sample correction. This is
where the measure differs from [`RollingSkew`](RollingSkew.md), which centers
the window on its mean and applies the adjusted Fisher-Pearson correction. For
high-frequency returns the mean is negligible next to their spread and the two
are close; on returns with a drift they are not. The value does not change
when the returns are scaled.

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
    from screamer import RollingRealizedSkew

    rng = np.random.default_rng(5)
    r = rng.standard_normal(78 * 40) * 0.001             # 40 days of 5-minute returns
    crash = slice(78 * 20, 78 * 22)
    r[crash] -= np.abs(rng.standard_normal(78 * 2)) ** 3 * 0.001   # a run of large drops
    skew = RollingRealizedSkew(78)(r)

    fig = go.Figure()
    fig.add_trace(go.Scatter(y=skew, name='realized skewness, 1-day window', line=dict(color='navy')))
    fig.update_layout(title='RollingRealizedSkew: large drops turn the skewness negative',
                      xaxis_title='5-minute step', yaxis_title='skewness',
                      margin=dict(l=20, r=20, t=60, b=20),
                      legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1))
    fig.show()
```

<!-- HELP_END -->
