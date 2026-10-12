---
name: EwKurt
title: Exponentially-weighted kurtosis
implementation_family: ew
topics:
- statistics
tags:
- ew
- kurtosis
- excess kurtosis
- fat tails
- exponential
- fourth moment
short: EW excess kurtosis.
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
- EwSkew
- EwStd
- RollingKurt
- ExpandingKurt
nan_policy: ignore
---

# `EwKurt`

## Description

`EwKurt` computes the exponentially weighted excess kurtosis of the input. The smoothing factor `alpha` is set by exactly one of `com`, `span`, `halflife` or `alpha`. It applies the adjusted excess-kurtosis estimator (the one scipy returns with `bias=False` and pandas returns from `rolling().kurt()`) to exponentially weighted moments, with the effective sample size $N_\text{eff}$ in place of the sample count.

Each step decays the running sums by $(1 - \alpha)$ and adds the new sample. $S_{x^k}$ stands for each power sum the statistic uses: $S_x$ for $k = 1$, $S_{xx}$ for $k = 2$, and so on. The implementation takes the sums about an origin close to the mean, which gives the same result without losing precision when the mean is large next to the spread (see [Accuracy at large levels](../conventions.md#accuracy-at-large-levels)).

$$
S_w \leftarrow (1-\alpha) S_w + 1, \qquad
S_{ww} \leftarrow (1-\alpha)^2 S_{ww} + 1, \qquad
S_{x^k} \leftarrow (1-\alpha) S_{x^k} + x_t^k
$$

From these sums, with $\bar{x} = S_x / S_w$ and the weighted central moments

$$
m_2 = \frac{S_{xx}}{S_w} - \bar{x}^2, \qquad
m_4 = \frac{S_{xxxx}}{S_w} - 4\bar{x}\frac{S_{xxx}}{S_w} + 6\bar{x}^2\frac{S_{xx}}{S_w} - 3\bar{x}^4, \qquad
N_\text{eff} = \frac{S_w^2}{S_{ww}},
$$

the output is

$$
\text{EwKurt}[t] = \frac{N_\text{eff} - 1}{(N_\text{eff} - 2)(N_\text{eff} - 3)}
\left( (N_\text{eff} + 1)\frac{m_4}{m_2^2} - 3(N_\text{eff} - 1) \right)
$$

The output is `NaN` while $N_\text{eff} \le 3$.

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

    np.random.seed(0)  # reproducible example data

    from screamer import EwKurt

    data = np.cumsum(np.random.normal(size=300))
    ewkurt_data = EwKurt(span=20)(data)

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[1/2, 1/2], vertical_spacing=0.1)

    fig.add_trace(go.Scatter(y=data, mode='lines', name='Original Data'), row=1, col=1)
    fig.add_trace(go.Scatter(y=ewkurt_data, mode='lines', name='EwKurt', line=dict(color='red')), row=2, col=1)

    fig.update_layout(
        title="Exponentially Weighted Moving Kurtosis",
        xaxis_title="Index",
        yaxis=dict(title="Original Data"),
        yaxis2=dict(title="EwKurt", range=[-1, 2]),
        margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    fig.show()
```

<!-- HELP_END -->
