# `ButterHighpass`

## Description

Digital high-pass Butterworth IIR filter, designed via the standard analog-prototype + `lp2hp` + bilinear pipeline.

$$
\text{ButterHighpass}(N, W_c)
$$

- `order` (int, $\ge 1$): filter order. Higher order = steeper roll-off, larger group delay.
- `cutoff_freq` (float in $(0, 1)$): normalised cutoff. `1` is the Nyquist frequency (matches `scipy.signal.butter`'s `Wn` convention).

## Implementation Details

Bit-exact match to `scipy.signal.butter(order, cutoff, btype='highpass')` + `scipy.signal.lfilter`. Verified to ~1e-12 in `tests/test_signal.py`.

```python
from screamer import ButterHighpass
hp = ButterHighpass(order=4, cutoff_freq=0.1)
out = hp(noisy_signal)
```

Uses the same `IIRFilter` engine as the existing `Butter` low-pass class -- O(order) per step.
