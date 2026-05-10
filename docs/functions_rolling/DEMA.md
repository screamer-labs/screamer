# `DEMA`

## Description

`DEMA` (Double Exponential Moving Average, Patrick Mulloy 1994) is a low-lag smoother defined as:

$$
\text{DEMA}[t] = 2 \cdot \text{EMA}(x)[t] - \text{EMA}(\text{EMA}(x))[t]
$$

The construction subtracts the *lag* of a single EMA from twice that EMA. The result tracks the input more closely than a plain EMA of the same span, while still smoothing high-frequency noise.

## Parameters

Same `com / span / halflife / alpha` mutex as `EwMean` -- specify exactly one. The same value is used for both internal EMAs.

*NaN handling*: NaN values should be preprocessed.

## Implementation Details

### Algorithm

`DEMA` is a pure composition of two chained `EwMean` instances. The class holds them as members and combines their outputs as `2*e1 - e2`. There is no warmup: each `EwMean` returns a valid value from t=0 (`sum_x / sum_w` is `x[0]/1 = x[0]` after one append), so `DEMA[0] = 2*x[0] - x[0] = x[0]`.

### Complexity

* Time complexity: `O(1)` per step.
* Space complexity: `O(1)`.

## Usage Example

```python
import numpy as np
from screamer import DEMA, EwMean

x = np.cumsum(np.random.randn(100))

# Direct
ours = DEMA(span=10)(x)

# Algorithmically equivalent composition (the test suite verifies bit-equality)
e1 = EwMean(span=10)(x)
e2 = EwMean(span=10)(e1)
np.testing.assert_allclose(ours, 2*e1 - e2, atol=1e-12)
```

## Reference

Equivalent to TA-Lib's `DEMA`. Validated in `tests/test_moving_averages.py` against the explicit `2*EwMean - EwMean(EwMean)` composition for four span values.
