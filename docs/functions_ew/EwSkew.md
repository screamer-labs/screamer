---
name: EwSkew
title: Exponentially-weighted skewness
implementation_family: ew
topics:
- statistics
tags:
- ew
- skewness
- skew
- exponential
- third moment
short: EW skewness.
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
- EwKurt
- EwStd
- RollingSkew
- ExpandingSkew
nan_policy: ignore
---

# `EwSkew`

## Description

`EwSkew` computes the exponentially weighted skewness of the input. The smoothing factor `alpha` is set by exactly one of `com`, `span`, `halflife` or `alpha`. It applies the adjusted Fisher-Pearson estimator (the one scipy returns with `bias=False` and pandas returns from `rolling().skew()`) to exponentially weighted moments, with the effective sample size $N_\text{eff}$ in place of the sample count.

Each step decays the running sums by $(1 - \alpha)$ and adds the new sample. $S_{x^k}$ stands for each power sum the statistic uses: $S_x$ for $k = 1$, $S_{xx}$ for $k = 2$, and so on. The implementation takes the sums about an origin close to the mean, which gives the same result without losing precision when the mean is large next to the spread (see [Accuracy at large levels](../conventions.md#accuracy-at-large-levels)).

$$
S_w \leftarrow (1-\alpha) S_w + 1, \qquad
S_{ww} \leftarrow (1-\alpha)^2 S_{ww} + 1, \qquad
S_{x^k} \leftarrow (1-\alpha) S_{x^k} + x_t^k
$$

From these sums, with $\bar{x} = S_x / S_w$ and the weighted central moments

$$
m_2 = \frac{S_{xx}}{S_w} - \bar{x}^2, \qquad
m_3 = \frac{S_{xxx}}{S_w} - 3\bar{x}\frac{S_{xx}}{S_w} + 2\bar{x}^3, \qquad
N_\text{eff} = \frac{S_w^2}{S_{ww}},
$$

the output is

$$
\text{EwSkew}[t] = \frac{\sqrt{N_\text{eff}(N_\text{eff} - 1)}}{N_\text{eff} - 2} \cdot \frac{m_3}{m_2^{3/2}}
$$

The output is `NaN` while $N_\text{eff} \le 2$. It is also `NaN` once the series has stopped varying, after roughly 45 / alpha rows, and for a while after a single extreme value.

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
    from screamer import EwSkew

    np.random.seed(0)  # reproducible example data

    data = np.cumsum(np.random.normal(size=300))
    ewskew_data = EwSkew(span=20)(data)

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[1/2, 1/2],
        vertical_spacing=0.1
    )

    fig.add_trace(go.Scatter(y=data, mode='lines', name='Original Data'), row=1, col=1)
    fig.add_trace(go.Scatter(y=ewskew_data, mode='lines', name='EwSkew', line=dict(color='red')), row=2, col=1)

    fig.update_layout(
        title="Exponentially Weighted Moving Skewness",
        xaxis_title="Index",
        yaxis=dict(title="Original Data"),
        yaxis2=dict(title="EwSkew", range=[-.5, .5]),
        margin=dict(l=20, r=20, t=80, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)        
    )

    fig.show()
```

<!-- HELP_END -->
