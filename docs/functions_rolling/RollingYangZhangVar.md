---
name: RollingYangZhangVar
title: Rolling Yang-Zhang varariance
implementation_family: rolling
topics:
- volatility
tags:
- yang-zhang
- range-based
- ohlc
- drift-robust
- gap-aware
- var
- rolling
short: Var form of the Yang-Zhang estimator (drift + gap robust).
inputs: 4
outputs: 1
parameters:
- name: window_size
  type: int
  default: 20
  min: 2
  description: Smoothing window.
nan_policy: ignore
---

# `RollingYangZhangVar`

## Description

The Yang-Zhang (2000) estimator combines three variance components:

$$
\begin{aligned}
\sigma^2_o    &= \text{sample variance of overnight log returns } \ln(O_t / C_{t-1}) \\
\sigma^2_c    &= \text{sample variance of open-to-close log returns } \ln(C_t / O_t) \\
\sigma^2_{RS} &= \text{mean of per-bar Rogers-Satchell estimates} \\
k             &= \dfrac{0.34}{1.34 + (n+1)/(n-1)} \\
\sigma^2_{YZ} &= \sigma^2_o + k\ \cdot\ \sigma^2_c + (1-k)\ \cdot\ \sigma^2_{RS}
\end{aligned}
$$

The only classical estimator that handles **both** drift *and* overnight gaps. ~14x
efficient vs close-to-close.

**4-input, 1-output** on `(open, high, low, close)`. First valid output at sample index
`window_size` (we need n+1 price bars to form n overnight returns). The `Vol` variant
returns `sqrt(Var)` (bit-exact via the same internal state).

No EW form is exposed because the `k` factor depends on a discrete window size; any
"EW analogue" would require an arbitrary mapping from `span` to `n` that varies by
convention.


<!-- NAN_FOOTNOTE_START -->
## NaN handling

**Policy: `ignore`.** A `NaN` in any input at index `t` causes the function to skip that step: output at `t` is `NaN` and internal state is unchanged. Subsequent finite samples are processed as if step `t` had not occurred.
<!-- NAN_FOOTNOTE_END -->

<!-- HELP_END -->
