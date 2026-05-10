# `TEMA`

## Description

`TEMA` (Triple Exponential Moving Average, Patrick Mulloy 1994) extends the `DEMA` construction by one more level:

$$
\text{TEMA}[t] = 3 \cdot \text{EMA}(x)[t] - 3 \cdot \text{EMA}(\text{EMA}(x))[t] + \text{EMA}(\text{EMA}(\text{EMA}(x)))[t]
$$

The three-term combination further reduces lag: `TEMA` typically tracks faster than `DEMA` and much faster than a plain EMA of the same span, in exchange for slightly less smoothing.

## Parameters

Same `com / span / halflife / alpha` mutex as `EwMean` -- specify exactly one. The same value is used for all three internal EMAs.

*NaN handling*: NaN values should be preprocessed.

## Implementation Details

### Algorithm

Pure composition of three chained `EwMean` instances. No explicit warmup: each `EwMean` returns a valid value from t=0, so `TEMA[0] = 3*x[0] - 3*x[0] + x[0] = x[0]`.

### Complexity

* Time complexity: `O(1)` per step.
* Space complexity: `O(1)`.

## Usage Example

```python
import numpy as np
from screamer import TEMA, EwMean

x = np.cumsum(np.random.randn(100))

# Direct
ours = TEMA(span=10)(x)

# Algorithmically equivalent composition (the test suite verifies equality)
e1 = EwMean(span=10)(x)
e2 = EwMean(span=10)(e1)
e3 = EwMean(span=10)(e2)
np.testing.assert_allclose(ours, 3*e1 - 3*e2 + e3, atol=1e-12)
```

## Reference

Equivalent to TA-Lib's `TEMA`. Validated in `tests/test_moving_averages.py` against the explicit composition for four span values.
