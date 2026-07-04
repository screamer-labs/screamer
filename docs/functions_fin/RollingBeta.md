---
name: RollingBeta
title: Rolling beta
implementation_family: fin
topics:
- correlation
- regression
tags:
- beta
- regression
- capm
- pair
short: cov(x, y) / var(y) - regression slope of x on y.
inputs: 2
outputs: 1
parameters:
- name: window_size
  type: int
  default: 20
  min: 2
  description: Window length.
- name: start_policy
  type: str
  default: strict
  enum:
  - strict
  - expanding
  - zero
  description: Warmup behaviour.
nan_policy: ignore
---

# `RollingBeta`

## Description

`RollingBeta` computes the rolling regression slope of `x` on `y` over a sliding window of fixed size. The first argument is the dependent variable (target), the second is the independent variable (regressor).

This is the CAPM-style β: `RollingBeta(asset_returns, market_returns, w)` returns the asset's beta with respect to the market.

*Equation*:

$$
\beta_w[t] = \frac{\mathrm{cov}(x, y)}{\mathrm{var}(y)} = \frac{n \sum x_i y_i - \sum x_i \sum y_i}{n \sum y_i^2 - (\sum y_i)^2}
$$

with the sums taken over the most recent `window_size` samples.

*Parameters*:

- **`window_size`** (`int`, ≥ 2): size of the rolling window.
- **`start_policy`** (`str`, default `"strict"`): controls warmup behavior. See `RollingMean` for the full definition.

*Input shape*: two parallel streams, identical to [`RollingCorr`](RollingCorr.md).

*Return value*: the slope coefficient. Returns `NaN` during warmup or when `y` has zero variance within the window (regression is undefined).

> **Convention note**: pandas does not ship a `rolling().beta()` method directly. The pandas-equivalent expression is `pd.Series(x).rolling(w).cov(pd.Series(y)) / pd.Series(y).rolling(w).var()`. Some libraries call this slope "alpha" or use the inverse argument order; double-check the convention when comparing.


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

Four `detail::RollingSum` buffers maintain `Σx`, `Σy`, `Σyy`, `Σxy` over the window. Each new sample updates all four sums in `O(1)`, then β is computed in closed form.

* **Time**: `O(1)` per new element.
* **Space**: `O(window_size)` (four circular buffers).
* **Reference**: parity with `pd.Series(x).rolling(w).cov(pd.Series(y)) / pd.Series(y).rolling(w).var()` verified in `tests/test_rolling_two_input.py`.

### Related

- [`RollingCov`](RollingCov.md) is the numerator alone.
- [`RollingSpread`](RollingSpread.md) returns `x - β·y` using exactly the same β.
