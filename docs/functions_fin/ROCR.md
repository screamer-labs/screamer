# `ROCR`

## Description

`ROCR(k)` is the **rate of change** over `k` steps, expressed as a *ratio*:

$$
\text{ROCR}[t] = \frac{x[t]}{x[t-k]}
$$

The ratio form is convenient when you want to chain returns multiplicatively (e.g. cumulative-product wealth paths) without subtracting 1 each step.

*Parameters*:

- `window_size` (int, positive): the lookback `k`.

*NaN handling*: NaN for the first `k` samples; NaN when `x[t-k] == 0`.

## Identity to ROCP

`ROCR(k) - 1 == ROCP(k)` exactly. Pick whichever form keeps the calling code cleaner.

## Reference

Equivalent to `talib.ROCR(x, timeperiod=k)`. Bit-exact match (cross-validated in `tests/test_third_party_alignment.py`).
