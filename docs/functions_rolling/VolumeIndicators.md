# Volume-aware indicators

Five classic indicators that read **price + volume**: `RollingVWAP`, `OBV`, `AD`, `ADOSC`, `MFI`. They split into two families: simple volume aggregation (VWAP, OBV, AD) and volume-weighted oscillators (ADOSC, MFI).

## `RollingVWAP(window_size)` -- Volume-weighted average price

$$
\text{TP}[t]   = (\text{high} + \text{low} + \text{close}) / 3 \qquad
\text{VWAP}[t] = \dfrac{\sum_w \text{TP} \cdot \text{volume}}{\sum_w \text{volume}}
$$

**4-input, 1-output** on `(high, low, close, volume)`. Uses the typical price as the weighting basis (matches pandas-ta-classic's `vwap`). For a *session-VWAP* (cumulative since some reset point), call `reset()` at the session boundary.

First valid output at sample index `window_size - 1`. Composes two `detail::RollingSum` instances; O(1) per step.

## `OBV()` -- On-Balance Volume (Granville, 1963)

$$
\text{OBV}[t] = \text{OBV}[t-1] + \text{sign}(C - C_{t-1}) \cdot V[t]
$$

with seed `OBV[0] = volume[0]` (TA-Lib's convention). **2-input, 1-output** on `(close, volume)`. Cumulative; no window. Bit-exact to `talib.OBV` (0.0 difference).

## `AD()` -- Accumulation / Distribution Line (Chaikin)

$$
\text{CLV}[t] = \dfrac{(C - L) - (H - C)}{H - L} \qquad
\text{AD}[t]  = \text{AD}[t-1] + \text{CLV} \cdot V[t]
$$

The "close location value" is in `[-1, +1]`: +1 means the close is at the bar's high (full accumulation), -1 at the low (full distribution). When `high == low` the CLV is undefined and the AD line is unchanged (TA-Lib's convention).

**4-input, 1-output** on `(high, low, close, volume)`. Cumulative; no window. Bit-exact to `talib.AD`.

## `ADOSC(fast=3, slow=10)` -- Chaikin A/D Oscillator

$$
\text{ADOSC}[t] = \text{EMA}(\text{AD},\ \text{fast})[t] - \text{EMA}(\text{AD},\ \text{slow})[t]
$$

Difference of two EMAs of the A/D line. **4-input, 1-output** on `(high, low, close, volume)`. Default `(3, 10)` matches TA-Lib.

The underlying EMA is `screamer.EwMean` (pandas `adjust=True`), so `ADOSC` inherits the same documented divergence from TA-Lib's `ADOSC` as `DEMA` / `TEMA` / `MACD` / `TRIX`. The class matches the explicit pandas-composition reference bit-exactly. See [conventions](../conventions.md) for the divergence detail.

## `MFI(window_size=14)` -- Money Flow Index

Volume-weighted analogue of RSI on the typical price:

$$
\begin{aligned}
\text{TP}[t]     &= (H + L + C) / 3 \\
\text{MF}[t]     &= \text{TP} \cdot V \\
\text{pos\_MF}_w &= \sum_w \text{MF}[\text{where}\ \text{TP} > \text{TP}_{t-1}] \\
\text{neg\_MF}_w &= \sum_w \text{MF}[\text{where}\ \text{TP} < \text{TP}_{t-1}] \\
\text{MFI}[t]    &= 100 \cdot \dfrac{\text{pos\_MF}_w}{\text{pos\_MF}_w + \text{neg\_MF}_w}
\end{aligned}
$$

**4-input, 1-output** on `(high, low, close, volume)`. First valid at sample index `window_size`. Bit-exact to `talib.MFI` (~1e-14).

## Usage Example

```python
import numpy as np
from screamer import RollingVWAP, OBV, AD, ADOSC, MFI

rng = np.random.default_rng(0)
n = 200
close = 100 + np.cumsum(rng.normal(0, 1, n))
high = close + np.abs(rng.normal(0, 0.5, n))
low = close - np.abs(rng.normal(0, 0.5, n))
volume = 1000 + np.abs(rng.normal(0, 200, n))

vwap  = RollingVWAP(20)(high, low, close, volume)
obv   = OBV()(close, volume)
ad    = AD()(high, low, close, volume)
adosc = ADOSC(3, 10)(high, low, close, volume)
mfi   = MFI(14)(high, low, close, volume)
```

## Reference table

| Indicator | TA-Lib match |
|---|---|
| `RollingVWAP` | No TA-Lib counterpart; bit-exact to pandas composition |
| `OBV` | bit-exact (`0.0`) |
| `AD` | bit-exact (`0.0`) |
| `ADOSC` | **divergent** by EMA convention; bit-exact to pandas composition |
| `MFI` | bit-exact (`~1e-14`) |
