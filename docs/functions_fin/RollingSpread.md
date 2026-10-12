---
name: RollingSpread
title: Rolling spread
implementation_family: fin
topics:
- regression
- pairs
tags:
- spread
- hedge
- hedge ratio
- residual
- pair
- pairs trading
short: Rolling hedge-adjusted residual x - beta * y, with beta the rolling regression slope.
inputs: 2
outputs: 1
parameters:
- name: window_size
  type: int
  default: 20
  min: 2
  description: Number of most recent (x, y) pairs used to estimate the hedge ratio.
- name: start_policy
  type: str
  default: strict
  enum:
  - strict
  - expanding
  - zero
  description: 'Output before the window is full: "strict" gives NaN, "expanding" computes
    from the samples seen so far, and "zero" treats the window as pre-filled with zeros.'
nan_policy: ignore
see_also:
- RollingBeta
- RollingResidualStd
- RollingZscore
- EwSpread
- RollingTLSBeta
---

# `RollingSpread`

## Description

`RollingSpread` computes the rolling hedge-adjusted residual of `x` against `y`. At each time step it computes the rolling regression slope `β_w[t] = cov(x, y) / var(y)` over the window and returns `x[t] - β_w[t] · y[t]`.

$$
\beta_w[t] = \frac{\mathrm{cov}(x, y)}{\mathrm{var}(y)},
\qquad
\mathrm{spread}_w[t] = x[t] - \beta_w[t]\, y[t]
$$

with the sums in `cov` and `var` taken over the most recent `window_size` samples ending at `t`.

It takes two parallel streams, like [`RollingCorr`](RollingCorr.md). The output is `NaN` during warmup and when `y` has zero variance within the window.


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
    from plotly.subplots import make_subplots
    from screamer import RollingSpread

    np.random.seed(0)
    N = 400
    common = np.cumsum(np.random.normal(size=N))
    a = common + 0.3 * np.cumsum(np.random.normal(size=N))
    b = 1.2 * common + 0.3 * np.cumsum(np.random.normal(size=N))
    # Inject a transient mispricing in a between samples 200 and 240.
    a[200:240] += 5.0

    spread_60 = RollingSpread(window_size=60)(a, b)

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.5, 0.5], vertical_spacing=0.05)
    fig.add_trace(go.Scatter(y=a, mode="lines", name="a"), row=1, col=1)
    fig.add_trace(go.Scatter(y=b, mode="lines", name="b"), row=1, col=1)
    fig.add_trace(go.Scatter(y=spread_60, mode="lines",
                             name="RollingSpread(60)"), row=2, col=1)
    fig.update_layout(
        title="Hedge-adjusted spread of a against b (window=60)",
        margin=dict(l=20, r=20, t=60, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_yaxes(title_text="price", row=1, col=1)
    fig.update_yaxes(title_text="spread", row=2, col=1)
    fig.show()
```

<!-- HELP_END -->

## Implementation Details

It keeps the same running window sums as [`RollingBeta`](RollingBeta.md). The spread at each step combines the rolling β with the current `(x, y)` pair.

* **Time**: `O(1)` per new element.
* **Space**: `O(window_size)`.
* **Reference**: parity with `x - RollingBeta(x, y, w) · y` verified in `tests/test_rolling_two_input.py`.

[`RollingZscore`](../functions_rolling/RollingZscore.md) applied to the spread gives the usual pairs-trading mean-reversion signal.
