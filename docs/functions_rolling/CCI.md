---
name: CCI
title: Commodity Channel Index (CCI)
implementation_family: rolling
topics:
- momentum
tags:
- cci
- lambert
- oscillator
- talib
- hlc
short: Commodity Channel Index over typical price.
inputs: 3
outputs: 1
parameters:
- name: window_size
  type: int
  default: 14
  min: 2
  description: Period (Wilder's default).
nan_policy: ignore
---

# `CCI`

## Description

`CCI` (Commodity Channel Index, Donald Lambert) measures how far the current bar's typical price has moved from its rolling mean, normalised by the mean absolute deviation of the same window:

$$
\begin{aligned}
\text{TP}[t]      &= (\text{high} + \text{low} + \text{close}) / 3 \\
\overline{\text{TP}} &= \text{SMA}(\text{TP},\ n) \\
\text{MAD}        &= \text{mean}\big(\ |\text{TP} - \overline{\text{TP}}|\ \big) \quad \text{over the same window} \\
\text{CCI}[t]     &= \frac{\text{TP}[t] - \overline{\text{TP}}[t]}{0.015 \cdot \text{MAD}[t]}
\end{aligned}
$$

The 0.015 constant is a Lambert convention: roughly 70-80% of CCI readings fall in `[-100, +100]` for a normal-distributed input.

**3-input, 1-output** (`FunctorBase<_, 3, 1>`) on `(high, low, close)`.

*Parameters*:

- `window_size` (int, default `14`).

*Warmup*: NaN for the first `window_size - 1` samples.

*Range-zero*: returns `0` when the MAD is 0 (a perfectly flat window).

*NaN handling*: NaN inputs poison the rolling sum.


<!-- NAN_FOOTNOTE_START -->
## NaN handling

**Policy: `ignore`.** A `NaN` in any input at index `t` causes the function to skip that step: output at `t` is `NaN` and internal state is unchanged. Subsequent finite samples are processed as if step `t` had not occurred.
<!-- NAN_FOOTNOTE_END -->

<!-- HELP_END -->

## Implementation Details

Holds one circular buffer of TP values with a rolling sum (the SMA part is incremental O(1)); the per-step MAD sweep over the window is `O(window_size)`. Same trade-off as `RollingMad`, and the same reason: there is no closed-form O(1) MAD when the mean shifts each step.

## Reference

Bit-exact match to `talib.CCI(high, low, close, timeperiod)` post-warmup (verified to ~1e-11 in `tests/test_third_party_alignment.py`).
