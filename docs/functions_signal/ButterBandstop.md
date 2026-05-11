# `ButterBandstop`

## Description

Digital band-stop (notch) Butterworth IIR filter -- suppresses frequencies in `[low_cutoff, high_cutoff]`. Produces a 2N-order filter from the order-N analog prototype.

$$
\text{ButterBandstop}(N, W_{\text{low}}, W_{\text{high}})
$$

- `order` (int, $\ge 1$): order of the analog prototype.
- `low_cutoff`, `high_cutoff` (floats in $(0, 1)$, with `low < high`): stop band as fractions of the Nyquist frequency.

Bit-exact match to `scipy.signal.butter(order, [low, high], btype='bandstop')` + `scipy.signal.lfilter`. Verified to ~1e-9 in `tests/test_signal.py`.

```python
from screamer import ButterBandstop
# 60 Hz notch on 1kHz-sampled data: Nyquist = 500 Hz, cutoffs ~58/500 .. 62/500
notch = ButterBandstop(order=2, low_cutoff=0.116, high_cutoff=0.124)
out = notch(signal)
```
