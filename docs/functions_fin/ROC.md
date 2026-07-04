---
name: ROC
title: Rate of change (ROC)
implementation_family: fin
topics:
- momentum
tags:
- roc
- rate-of-change
- talib
short: 100 * (x[t] / x[t-k] - 1) - TA-Lib's ROC.
inputs: 1
outputs: 1
parameters:
- name: window_size
  type: int
  default: 10
  min: 1
  description: Lookback k. TA-Lib default is 10.
nan_policy: propagate
---

# `ROC`

## Description

`ROC(k)` is the **rate of change** over `k` steps, expressed as a percentage:

$$
\text{ROC}[t] = 100 \cdot \frac{x[t] - x[t-k]}{x[t-k]}
$$

The three rate-of-change variants in this library, all matching their TA-Lib namesakes:

| Class | Formula | Returns |
|---|---|---|
| **`ROC`** | $100 \cdot (x[t] - x[t-k]) / x[t-k]$ | percentage (e.g. `5.0` = +5%) |
| `ROCP` | $(x[t] - x[t-k]) / x[t-k]$ | fraction (e.g. `0.05` = +5%); same as [`Return`](Return.md) |
| `ROCR` | $x[t] / x[t-k]$ | ratio (e.g. `1.05`) |

*Parameters*:

- `window_size` (int, positive): the lookback `k`.

*NaN handling*: NaN for the first `k` samples (no `x[t-k]` yet). Also NaN if `x[t-k] == 0` (division by zero).


<!-- NAN_FOOTNOTE_START -->
## NaN handling

**Policy: `propagate`.** Input `NaN` values are stored in the lookback. Output is `NaN` at any index where the function's positional formula references a `NaN` input; recovery happens once the `NaN` slides out of the lookback.
<!-- NAN_FOOTNOTE_END -->

<!-- HELP_END -->

## Reference

Equivalent to `talib.ROC(x, timeperiod=k)`. Bit-exact match to ~1e-14 (cross-validated in `tests/test_third_party_alignment.py`).
