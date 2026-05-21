---
name: RollingSpread
title: Rolling spread
implementation_family: fin
topics:
- correlation
- regression
tags:
- spread
- hedge
- pair
short: x - beta(x,y) * y — hedge-adjusted residual.
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

# `RollingSpread`

## Description

`RollingSpread` computes the rolling hedge-adjusted residual of `x` against `y`. At each time step it computes the rolling regression slope `β_w[t] = cov(x, y) / var(y)` over the window and returns `x[t] - β_w[t] · y[t]`.

This is the building block for pairs trading: `RollingSpread(price_a, price_b, w)` is the residual of `price_a` after removing its rolling-best-fit linear exposure to `price_b`. A mean-reverting spread is the prototypical pairs-trading signal.

*Equation*:

$$
\beta_w[t] = \frac{\mathrm{cov}(x, y)}{\mathrm{var}(y)},
\qquad
\mathrm{spread}_w[t] = x[t] - \beta_w[t]\, y[t]
$$

with the sums in `cov` and `var` taken over the most recent `window_size` samples ending at `t`.

*Parameters*:

- **`window_size`** (`int`, ≥ 2): size of the rolling window.
- **`start_policy`** (`str`, default `"strict"`): controls warmup behavior.

*Input shape*: two parallel streams, identical to [`RollingCorr`](RollingCorr.md).

*Return value*: a real number, the residual. Returns `NaN` during warmup or when `y` has zero variance within the window.


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

Same four `detail::RollingSum` buffers as [`RollingBeta`](RollingBeta.md), `O(1)` per step. The spread at each step combines the rolling β with the current `(x, y)` pair.

* **Time**: `O(1)` per new element.
* **Space**: `O(window_size)`.
* **Reference**: parity with `x - RollingBeta(x, y, w) · y` verified in `tests/test_rolling_two_input.py`.

### Related

- [`RollingBeta`](RollingBeta.md) returns the β alone (no residual).
- [`RollingZscore`](../functions_rolling/RollingZscore.md) applied to the spread output gives the standard pairs-trading mean-reversion signal.
