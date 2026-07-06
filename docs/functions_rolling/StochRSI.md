---
name: StochRSI
title: Stochastic RSI
implementation_family: rolling
topics:
- momentum
tags:
- stochrsi
- rsi
- oscillator
- talib
short: Stochastic applied to RSI (1 input -> 2 outputs).
inputs: 1
outputs: 2
parameters:
- name: rsi_period
  type: int
  default: 14
  min: 2
  description: RSI period.
- name: stoch_period
  type: int
  default: 14
  min: 2
  description: Stochastic lookback over the RSI.
- name: smooth_k
  type: int
  default: 1
  min: 1
  description: '%K smoothing.'
- name: d
  type: int
  default: 3
  min: 1
  description: '%D period.'
nan_policy: ignore
---

# `StochRSI`

## Description

`StochRSI` (Chande & Kroll, 1994) applies the Stochastic oscillator formula to an RSI series rather than to price. It is a "rate of change of momentum" indicator: more responsive than RSI alone and useful for spotting RSI turning points.

$$
\begin{aligned}
\text{RSI}[t]   &= \text{RollingRSI}(\text{rsi\_period},\ \text{method}=\text{wilder})(x)[t] \\
\text{raw\_K}   &= 100 \cdot \frac{\text{RSI} - \min(\text{RSI},\ \text{stoch\_period})}{\max(\text{RSI},\ \text{stoch\_period}) - \min(\text{RSI},\ \text{stoch\_period})} \\
\%K[t]          &= \text{SMA}(\text{raw\_K},\ \text{smooth\_k}) \\
\%D[t]          &= \text{SMA}(\%K,\ d)
\end{aligned}
$$

**1-input, 2-output** (`FunctorBase<_, 1, 2>`).

## Setting it up

| Configuration | Constructor | TA-Lib equivalent |
|---|---|---|
| **Fast StochRSI** (TA-Lib's `STOCHRSI`) -- default | `StochRSI(14, 14, 1, 3)` | `STOCHRSI(close, timeperiod=14, fastk_period=14, fastd_period=3)` |
| **Slow StochRSI** -- add smoothing on K | `StochRSI(14, 14, 3, 3)` | not in TA-Lib; common in pandas-ta `stochrsi(..., k=3, ...)` |

`smooth_k=1` is the identity SMA, collapsing `%K = raw_K` (the "fast" form). `smooth_k >= 2` gives the slow form.

## Parameters

- `rsi_period` (int, default `14`): RSI lookback. Uses Wilder smoothing internally.
- `stoch_period` (int, default `14`): rolling min/max window applied to the RSI series.
- `smooth_k` (int, default `1`): SMA period applied to `raw_K`.
- `d` (int, default `3`): SMA period applied to `%K` to produce `%D`.

*Warmup*: both outputs are NaN until `%D` is valid, at sample index `rsi_period + stoch_period + smooth_k + d - 3` (TA-Lib's convention -- gate both K and D together). For defaults that is index 29.

*Range-zero*: when the RSI window is flat (`max == min`), `raw_K` is undefined; we return `0`.


<!-- NAN_FOOTNOTE_START -->
## NaN handling

**Policy: `ignore`.** A `NaN` in any input at index `t` causes the function to skip that step: output at `t` is `NaN` and internal state is unchanged. Subsequent finite samples are processed as if step `t` had not occurred.
<!-- NAN_FOOTNOTE_END -->

<!-- HELP_END -->

## Implementation Details

Composition: an internal `RollingRSI(rsi_period, method="wilder")`, two `detail::MonotonicDeque` (for rolling RSI min / max), and two `detail::RollingMean` (for the smooth_k and d SMAs). Amortised O(1) per step.

## Reference

Matches `talib.STOCHRSI(close, timeperiod=N, fastk_period=K, fastd_period=D)` bit-exactly (to ~1e-13 in `tests/test_third_party_alignment.py`).
