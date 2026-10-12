---
name: RollingBeta
title: Rolling beta
implementation_family: fin
topics:
- regression
tags:
- beta
- hedge ratio
- regression slope
- regression
- capm
- pair
short: Rolling regression slope cov(x, y) / var(y) of the first input x (dependent) on the
  second input y (regressor).
inputs: 2
outputs: 1
parameters:
- name: window_size
  type: int
  default: 20
  min: 2
  description: Number of most recent (target, regressor) pairs in the regression window.
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
- RollingAlpha
- RollingCov
- RollingCorr
- RollingSpread
- RollingTLSBeta
---

# `RollingBeta`

## Description

`RollingBeta(window_size)(x, y)` computes the least-squares slope of `x` regressed on `y` over the last `window_size` samples. The first input `x` is the dependent series (target) and the second input `y` is the regressor, so `RollingBeta(w)(stock, market)` gives the market beta of the stock.

*Equation*:

$$
\beta_w[t] = \frac{\mathrm{cov}(x, y)}{\mathrm{var}(y)} = \frac{n \sum x_i y_i - \sum x_i \sum y_i}{n \sum y_i^2 - (\sum y_i)^2}
$$

with the sums taken over the most recent `window_size` samples.

It takes two parallel streams, like [`RollingCorr`](RollingCorr.md), and returns the slope coefficient. The output is `NaN` during warmup and when `y` has zero variance within the window, where the regression is undefined.

pandas has no rolling beta method. The equivalent pandas expression is `pd.Series(x).rolling(w).cov(pd.Series(y)) / pd.Series(y).rolling(w).var()`. Other libraries may put the regressor first, so check the argument order when comparing.


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
    from screamer import RollingBeta

    np.random.seed(0)
    N = 300
    market_returns = np.random.normal(size=N)
    # An asset whose beta to the market drifts from 0.5 to 1.5 over time.
    true_beta = np.linspace(0.5, 1.5, N)
    asset_returns = true_beta * market_returns + 0.3 * np.random.normal(size=N)

    beta_60 = RollingBeta(window_size=60)(asset_returns, market_returns)

    fig = make_subplots(rows=1, cols=1)
    fig.add_trace(go.Scatter(y=true_beta, mode="lines", name="true beta",
                             line=dict(color="grey", dash="dash")))
    fig.add_trace(go.Scatter(y=beta_60, mode="lines", name="RollingBeta(60)"))
    fig.update_layout(
        title="Rolling beta of asset returns on market returns",
        xaxis_title="time",
        yaxis_title="beta",
        margin=dict(l=20, r=20, t=60, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.show()
```

<!-- HELP_END -->

## Implementation Details

Four running window sums maintain `Σx`, `Σy`, `Σyy`, `Σxy` over the window, with `x` and `y` measured from origins close to their window means (see [Accuracy at large levels](../conventions.md#accuracy-at-large-levels)). Each new sample updates all four sums in `O(1)`, then β is computed in closed form.

* **Time**: `O(1)` per new element.
* **Space**: `O(window_size)`.
* **Reference**: parity with `pd.Series(x).rolling(w).cov(pd.Series(y)) / pd.Series(y).rolling(w).var()` verified in `tests/test_rolling_two_input.py`.

[`RollingCov`](RollingCov.md) is the numerator alone, and [`RollingSpread`](RollingSpread.md) returns `x - β·y` with the same β.
