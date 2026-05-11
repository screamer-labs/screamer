---
name: RollingRank
title: Rolling rank
implementation_family: fin
topics:
- statistics
tags:
- rank
- position
- pandas
short: Rank of the current value within the trailing window (1-based, average tie
  rule).
inputs: 1
outputs: 1
parameters:
- name: window_size
  type: int
  default: 20
  min: 2
  description: Trailing-window length.
---

# `RollingRank`

## Description

Where does the current value sit within the trailing window?

$$
\text{rank}[t] = (\text{\#values} < y_t) + 1 + \tfrac{1}{2}(\text{\#ties} - 1)
$$

Pandas's "average" tie-breaking rule (mean rank among tied values). Returns a 1-based rank
in `[1, w]`.

1→1. Circular window buffer + per-step counting sweep; O(W) per step. Bit-exact (0.0) to
`pandas.Series.rolling(w).rank()`.

<!-- HELP_END -->
