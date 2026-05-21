---
name: RollingPercentile
title: Rolling percentile
implementation_family: fin
topics:
- statistics
tags:
- percentile
- position
- pandas
short: Percentile (rank/window) of the current value in the trailing window.
inputs: 1
outputs: 1
parameters:
- name: window_size
  type: int
  default: 20
  min: 2
  description: Trailing-window length.
nan_policy: ignore
---

# `RollingPercentile`

## Description

Percentile position of the current value within the trailing window:

$$
\text{percentile}[t] = \text{rank}[t] / w
$$

with the same "average" tie rule as `RollingRank`. Returns values in `[1/w, 1]`. Bit-exact
(0.0) to `pandas.Series.rolling(w).rank(pct=True)`.

1→1. O(W) per step.


<!-- NAN_FOOTNOTE_START -->
## NaN handling

**Policy: `ignore`.** A `NaN` in any input at index `t` causes the function to skip that step: output at `t` is `NaN` and internal state is unchanged. Subsequent finite samples are processed as if step `t` had not occurred.
<!-- NAN_FOOTNOTE_END -->

<!-- HELP_END -->
