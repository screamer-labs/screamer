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

## Reference

Equivalent to `talib.ROC(x, timeperiod=k)`. Bit-exact match to ~1e-14 (cross-validated in `tests/test_third_party_alignment.py`).
