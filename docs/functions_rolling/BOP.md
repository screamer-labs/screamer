---
name: BOP
title: Balance of Power (BOP)
implementation_family: rolling
topics:
- oscillator
tags:
- bop
- oscillator
- talib
- ohlc
short: (close - open) / (high - low) per bar. No smoothing.
inputs: 4
outputs: 1
parameters: []
---

# `BOP`

## Description

`BOP` (Balance of Power, Igor Livshin) measures whether a bar closes near its high (buyers in control) or its low (sellers in control):

$$
\text{BOP}[t] = \frac{\text{close} - \text{open}}{\text{high} - \text{low}}
$$

Output range is `[-1, +1]` for any sensibly-formed bar (where `low ≤ open, close ≤ high`).

**4-input, 1-output** (`FunctorBase<_, 4, 1>`). Argument order matches TA-Lib's `BOP`: `(open, high, low, close)`.

*Warmup*: none -- stateless, value defined for every input.

*Range-zero*: returns `0` when `high == low` (flat bar; convention matches TA-Lib).

<!-- HELP_END -->

## Implementation Details

Single per-step arithmetic operation; O(1) per step, zero state.

## Reference

Bit-exact match to `talib.BOP(open, high, low, close)` (verified to 0.0 -- exact integer-rounded arithmetic).
