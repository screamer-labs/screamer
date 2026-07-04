---
name: Diff2
title: Second difference
implementation_family: misc
topics:
- transforms
- momentum
tags:
- diff
- second-derivative
short: Second-order finite difference (discrete second derivative).
inputs: 1
outputs: 1
parameters:
- name: start_policy
  type: str
  default: strict
  enum:
  - strict
  - expanding
  - zero
  description: 'Warmup behaviour: ''strict'' (NaN until full window), ''expanding''
    (use partial windows), or ''zero'' (treat missing as zero).'
nan_policy: propagate
---

# `Diff2`

## Description

The `Diff2` function computes the *second-order* finite difference of the input - the discrete analogue of a second derivative. It is the result of applying a one-step difference twice.

*Equation*:

$$
y[t] = x[t] - 2 \, x[t-1] + x[t-2]
$$

equivalent to $\Delta(\Delta x)$.

*Parameters*:

- `start_policy` (str, optional): warmup behaviour, `"strict"` (default), `"truncate"`, or `"zero"`. Two warmup samples are needed before a valid output is produced.

*NaN handling*: Under `start_policy="strict"` the first two outputs are `NaN`. NaN inputs propagate.

*Note*: `Diff2` is **not** the same as `Diff(2)`. `Diff(2)` is the *lag-2 first* difference $x[t] - x[t-2]$. `Diff2` is the *second-order* difference, i.e. the difference of differences. On a quadratic input $x[t] = a t^2 + b t + c$ the result is the constant $2a$.


<!-- NAN_FOOTNOTE_START -->
## NaN handling

**Policy: `propagate`.** Input `NaN` values are stored in the lookback. Output is `NaN` at any index where the function's positional formula references a `NaN` input; recovery happens once the `NaN` slides out of the lookback.
<!-- NAN_FOOTNOTE_END -->

## Examples

### Usage example

```{eval-rst}
.. plotly::
    :include-source: True

    import numpy as np
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from screamer import Diff2

    n = 200
    t = np.arange(n)
    x = 0.0005 * (t - 100)**2 + np.sin(t / 10.0)
    out = Diff2()(x)

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.5, 0.5], vertical_spacing=0.1)
    fig.add_trace(go.Scatter(y=x, mode='lines', name='Input (parabola + sine)'),
                  row=1, col=1)
    fig.add_trace(go.Scatter(y=out, mode='lines',
                             name='Diff2 (discrete second derivative)',
                             line=dict(color='green')),
                  row=2, col=1)
    fig.update_layout(
        title="Diff2: Discrete Second Derivative",
        xaxis_title="Index",
        yaxis_title="Original",
        yaxis2_title="Diff2",
        margin=dict(l=20, r=20, t=60, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    fig.show()
```

<!-- HELP_END -->

## Reference

Equivalent to applying `numpy.diff` twice (with a warmup of two `NaN` samples preserved).
