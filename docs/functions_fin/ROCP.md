---
name: ROCP
title: Rate of change percent (ROCP)
implementation_family: fin
topics:
- momentum
tags:
- rocp
- rate-of-change
- talib
short: x[t] / x[t-k] - 1 — TA-Lib's ROCP. Identical to Return.
inputs: 1
outputs: 1
parameters:
- name: window_size
  type: int
  default: 10
  min: 1
  description: Lookback k.
---

# `ROCP`

## Description

`ROCP(k)` is the **rate of change** over `k` steps, expressed as a *fraction*:

$$
\text{ROCP}[t] = \frac{x[t] - x[t-k]}{x[t-k]}
$$

**This is exactly `Return(k)`.** It exists as a separately-named class because TA-Lib calls this indicator `ROCP`. Internally `ROCP` is a thin subclass of `Return` -- the implementation (delay buffer + subtract + divide) is shared, not duplicated.

| You want... | Use |
|---|---|
| TA-Lib parity (writing `ROCP` in ported code) | `ROCP(k)` |
| Any other context (returns workflows, log-returns family) | `Return(k)` |

*Parameters*:

- `window_size` (int, positive): the lookback `k`.

*NaN handling*: NaN for the first `k` samples; NaN when `x[t-k] == 0`.

## See also

- [`Return`](Return.md) -- same class, the documentation lives there.
- [`ROC`](ROC.md) -- `100 * ROCP` (percentage form).
- [`ROCR`](ROCR.md) -- `x[t] / x[t-k]` (ratio form).

<!-- HELP_END -->

## Reference

Equivalent to `talib.ROCP(x, timeperiod=k)`. Bit-exact match (cross-validated in `tests/test_third_party_alignment.py`).
