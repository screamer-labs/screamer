---
name: Momentum
title: Momentum
implementation_family: misc
topics:
- momentum
tags:
- momentum
- mom
- talib
short: x[t] - x[t-k], TA-Lib's MOM. Mathematically identical to Diff(k).
inputs: 1
outputs: 1
parameters:
- name: window_size
  type: int
  default: 10
  min: 1
  description: Lookback k. TA-Lib defaults to 10.
- name: start_policy
  type: str
  default: strict
  enum:
  - strict
  - expanding
  - zero
  description: 'Warmup behaviour: ''strict'' (NaN until full window), ''expanding''
    (use partial windows), or ''zero'' (treat missing as zero).'
nan_policy: propagate
---

# `Momentum`

## Description

`Momentum(k)` returns the *raw price displacement* over `k` steps:

$$
\text{Momentum}[t] = x[t] - x[t-k]
$$

**This is exactly `Diff(k)`.** It exists as a separately-named class because TA-Lib calls this indicator `MOM` and traders look for "momentum" in the API. Internally `Momentum` is a thin subclass of `Diff` -- the implementation (delay buffer + subtraction) is shared, not duplicated.

*Parameters*:

- `window_size` (int, positive): the lookback `k`.
- `start_policy` (str, optional): `"strict"` (default), `"expanding"`, or `"zero"`. Same semantics as `Diff`.

## When to use which

| You want... | Use |
|---|---|
| TA-Lib parity (writing `MOM` in published strategies) | `Momentum(k)` |
| Any other context | `Diff(k)` |

Both produce identical output bit-for-bit (matches `talib.MOM` to 0.0 -- exact integer arithmetic).


<!-- NAN_FOOTNOTE_START -->
## NaN handling

**Policy: `propagate`.** Input `NaN` values are stored in the lookback. Output is `NaN` at any index where the function's positional formula references a `NaN` input; recovery happens once the `NaN` slides out of the lookback.
<!-- NAN_FOOTNOTE_END -->

<!-- HELP_END -->

## Reference

See [`Diff`](Diff.md) for the full documentation. Equivalent to `numpy.diff(x, n=1, prepend=NaN)[::k]`-like indexing or TA-Lib's `MOM`.
