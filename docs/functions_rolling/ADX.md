# `ADX`

## Description

`ADX` (Average Directional Index, J. Welles Wilder Jr. 1978) measures **trend strength**, not direction. It is the canonical filter for "are we trending or chopping?" Returns the triple `(+DI, -DI, ADX)` per step:

$$
\begin{aligned}
\text{TR}[t]  &= \max(H - L,\ |H - C_{t-1}|,\ |L - C_{t-1}|) \\
+\text{DM}[t] &= H - C_{t-1}\quad\text{if}\quad H - C_{t-1} > C_{t-1} - L > 0,\ \text{else}\ 0 \\
-\text{DM}[t] &= C_{t-1} - L\quad\text{if}\quad C_{t-1} - L > H - C_{t-1} > 0,\ \text{else}\ 0 \\
\text{ATR}   &= \text{Wilder}(\text{TR},\ w) \\
+\text{DI}   &= 100 \cdot \text{Wilder}(+\text{DM},\ w) / \text{ATR} \\
-\text{DI}   &= 100 \cdot \text{Wilder}(-\text{DM},\ w) / \text{ATR} \\
\text{DX}    &= 100 \cdot |+\text{DI} - -\text{DI}| / (+\text{DI} + -\text{DI}) \\
\text{ADX}   &= \text{Wilder}(\text{DX},\ w)
\end{aligned}
$$

**3-input, 3-output** (`FunctorBase<_, 3, 3>`) on `(high, low, close)`. Outputs are `(out[..., 0]=+DI, out[..., 1]=-DI, out[..., 2]=ADX)`.

## Parameters and warmup

- `window_size` (int, default `14`, the Wilder convention).

| Output | First valid sample |
|---|---|
| `+DI` | `window_size` |
| `-DI` | `window_size` |
| `ADX` | `2 * window_size - 1` (double-Wilder warmup) |

For the default `window_size=14`, `+DI`/`-DI` start at sample 14 and `ADX` at sample 27. Matches TA-Lib's `PLUS_DI` / `MINUS_DI` / `ADX` bit-exactly.

## Convention note

TA-Lib's Wilder smoother for the DI/DM/TR triplet uses a slightly different seed than its ATR smoother: accumulate `w-1` values during warmup, then apply the recurrence at the `w`-th value (sum form). The ADX smoother itself uses the standard SMA-of-`w`-values seed (average form). `screamer.ADX` implements both conventions inline to match TA-Lib exactly; it does **not** share state with the existing `ATR` class.

## Usage

```python
import numpy as np
from screamer import ADX

rng = np.random.default_rng(0)
n = 300
close = 100 + np.cumsum(rng.normal(0, 1, n))
high = close + np.abs(rng.normal(0, 0.5, n))
low = close - np.abs(rng.normal(0, 0.5, n))

out = ADX(14)(high, low, close)
plus_di, minus_di, adx = out[:, 0], out[:, 1], out[:, 2]
# adx > 25 -> trending; adx < 20 -> ranging (Wilder's heuristic)
```

## Reference

Matches `talib.PLUS_DI`, `talib.MINUS_DI`, `talib.ADX` bit-exactly (verified to ~1e-14 in `tests/test_third_party_alignment.py`).
