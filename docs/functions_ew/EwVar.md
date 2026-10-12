---
name: EwVar
title: Exponentially-weighted variance
implementation_family: ew
topics:
- statistics
- volatility
tags:
- ew
- variance
- exponential
- ewm var
- realized variance
short: EW variance (pandas adjust=True bias-corrected).
inputs: 1
outputs: 1
parameters:
- name: com
  type: float
  default: null
  description: Center of mass c, giving alpha = 1 / (1 + c). Must be positive. Give exactly
    one of com, span, halflife and alpha.
- name: span
  type: float
  default: 20.0
  description: Span s in samples, giving alpha = 2 / (s + 1). Must be greater than 1. Give
    exactly one of com, span, halflife and alpha. The default shown is an example value; the
    constructor has no default.
- name: halflife
  type: float
  default: null
  description: Half-life h in samples, giving alpha = 1 - exp(-ln(2) / h). Must be positive.
    Give exactly one of com, span, halflife and alpha.
- name: alpha
  type: float
  default: null
  description: Smoothing factor, with 0 < alpha < 1; a larger alpha gives recent samples more
    weight. Give exactly one of com, span, halflife and alpha.
see_also:
- EwStd
- EwMean
- EwCov
- RollingVar
- ExpandingVar
nan_policy: ignore
---

# `EwVar`

## Description

`EwVar` computes the exponentially weighted variance of the input, bias-corrected with the effective sample size of the weights. The smoothing factor `alpha` is set by exactly one of `com`, `span`, `halflife` or `alpha`. The result matches pandas `ewm(adjust=True).var()`.

Each step decays the running sums by $(1 - \alpha)$ and adds the new sample. $S_{x^k}$ stands for each power sum the statistic uses: $S_x$ for $k = 1$, $S_{xx}$ for $k = 2$, and so on. The implementation takes the sums about an origin close to the mean, which gives the same result without losing precision when the mean is large next to the spread (see [Accuracy at large levels](../conventions.md#accuracy-at-large-levels)).

$$
S_w \leftarrow (1-\alpha) S_w + 1, \qquad
S_{ww} \leftarrow (1-\alpha)^2 S_{ww} + 1, \qquad
S_{x^k} \leftarrow (1-\alpha) S_{x^k} + x_t^k
$$

$$
N_\text{eff} = \frac{S_w^2}{S_{ww}}, \qquad
\text{EwVar}[t] = \left( \frac{S_{xx}}{S_w} - \left( \frac{S_x}{S_w} \right)^2 \right) \frac{N_\text{eff}}{N_\text{eff} - 1}
$$

$N_\text{eff}$ is the effective sample size of the exponential weights, and the factor $N_\text{eff} / (N_\text{eff} - 1)$ is the bias correction. The first output is `NaN`, because $N_\text{eff} = 1$ after one sample.

<!-- NAN_FOOTNOTE_START -->
## NaN handling

**Policy: `ignore`.** A `NaN` in any input gives a `NaN` output at that step. The step is skipped: it does not enter the window or update the internal state. See [NaN and warmup](../nan_and_warmup.md).
<!-- NAN_FOOTNOTE_END -->

## Examples

### Description

```{eval-rst}
.. plotly::
    :include-source: True

    import numpy as np
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from screamer import EwVar

    np.random.seed(0)  # reproducible example data

    # Generate example data
    data = np.cumsum(np.random.normal(size=300))

    # Compute exponentially weighted variance with a span of 20
    ewvar_data = EwVar(span=20)(data)

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[1/2, 1/2],
        vertical_spacing=0.1
    )

    fig.add_trace(go.Scatter(y=data, mode='lines', name='Original Data'), row=1, col=1)
    fig.add_trace(go.Scatter(y=ewvar_data, mode='lines', name='EwVar', line=dict(color='red')), row=2, col=1)

    fig.update_layout(
        title="Exponentially Weighted Moving Variance",
        xaxis_title="Index",
        yaxis=dict(title="Original Data"),
        yaxis2=dict(title="EwVar", range=[0, 25]),
        margin=dict(l=20, r=20, t=80, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)        
    )

    fig.show()
```

<!-- HELP_END -->
