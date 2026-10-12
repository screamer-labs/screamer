---
name: RollingCorr
title: Rolling correlation
implementation_family: fin
topics:
- regression
tags:
- correlation
- rolling correlation
- pearson
- pair
short: Rolling Pearson correlation of two parallel streams.
inputs: 2
outputs: 1
parameters:
- name: window_size
  type: int
  default: 20
  min: 2
  description: Number of most recent (x, y) pairs in the window.
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
- RollingCov
- RollingBeta
- EwCorr
- RollingSpread
---

# `RollingCorr`

## Description

`RollingCorr` computes the rolling Pearson correlation between two streams over a sliding window of fixed size.

*Equation*:

$$
\rho_w[t] = \frac{n \sum_{i} x_i y_i - \sum_{i} x_i \sum_{i} y_i}
                  {\sqrt{\bigl(n \sum_{i} x_i^2 - (\sum_{i} x_i)^2\bigr)\,\bigl(n \sum_{i} y_i^2 - (\sum_{i} y_i)^2\bigr)}}
$$

with the sums taken over the most recent `window_size` samples and `n = window_size`.

`RollingCorr` takes two parallel streams. Each call accepts two scalars, two 1-D arrays, two 2-D arrays paired column by column, two N-D arrays, two iterators, or one list of `(x, y)` pairs; the [polymorphic API](../polymorphic_api.md) page has the full table.

The output is a number in `[-1, 1]`. It is `NaN` during warmup and when either stream is constant within the window (zero variance). With `start_policy="expanding"` the first output is `NaN`, because a correlation needs at least two samples.


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
    from screamer import RollingCorr

    np.random.seed(0)
    N = 300
    # Two correlated random walks; correlation strengthens in the middle third.
    eps_a = np.random.normal(size=N)
    eps_b = np.random.normal(size=N)
    rho = np.where((np.arange(N) > N / 3) & (np.arange(N) < 2 * N / 3), 0.85, 0.15)
    a = np.cumsum(eps_a)
    b = np.cumsum(rho * eps_a + np.sqrt(1 - rho ** 2) * eps_b)

    corr_30 = RollingCorr(window_size=30)(np.diff(a), np.diff(b))
    corr_60 = RollingCorr(window_size=60)(np.diff(a), np.diff(b))

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.5, 0.5], vertical_spacing=0.05)
    fig.add_trace(go.Scatter(y=a, mode="lines", name="a"), row=1, col=1)
    fig.add_trace(go.Scatter(y=b, mode="lines", name="b"), row=1, col=1)
    fig.add_trace(go.Scatter(y=corr_30, mode="lines", name="RollingCorr(30)"), row=2, col=1)
    fig.add_trace(go.Scatter(y=corr_60, mode="lines", name="RollingCorr(60)"), row=2, col=1)
    fig.update_layout(
        title="RollingCorr on two random walks (windows 30 and 60)",
        margin=dict(l=20, r=20, t=60, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_yaxes(title_text="price", row=1, col=1)
    fig.update_yaxes(title_text="rolling correlation", range=[-1, 1], row=2, col=1)
    fig.show()
```

<!-- HELP_END -->

## Implementation Details

### Algorithm

Five running window sums maintain `Σx`, `Σy`, `Σxx`, `Σyy` and `Σxy` over the window, with `x` and `y` measured from origins close to their window means (see [Accuracy at large levels](../conventions.md#accuracy-at-large-levels)). Each new sample updates all five sums in `O(1)`, then the correlation is computed from the closed-form expression above.

### Complexity

* **Time**: `O(1)` per new element.
* **Space**: `O(window_size)`.

### pandas compatibility

Matches `pd.Series(x).rolling(w).corr(pd.Series(y))` to floating-point tolerance. Verified in `tests/test_rolling_corr.py`.
