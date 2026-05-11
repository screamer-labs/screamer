# `UltimateOscillator`

## Description

`UltimateOscillator` (Larry Williams, 1976) combines three timeframes of "buying pressure to true range" ratios into a single weighted oscillator. The three timeframes are intended to capture short, medium, and long momentum simultaneously.

$$
\begin{aligned}
\text{BP}[t]   &= \text{close} - \min(\text{low},\ \text{close}_{t-1}) \\
\text{TR}[t]   &= \max(\text{high},\ \text{close}_{t-1}) - \min(\text{low},\ \text{close}_{t-1}) \\
\text{avg}_k   &= \frac{\sum \text{BP} \text{ over } \text{period}_k}{\sum \text{TR} \text{ over } \text{period}_k} \\
\text{UO}[t]   &= 100 \cdot \frac{4 \cdot \text{avg}_1 + 2 \cdot \text{avg}_2 + \text{avg}_3}{7}
\end{aligned}
$$

The 4 / 2 / 1 weighting puts the heaviest emphasis on the shortest period.

**3-input, 1-output** (`FunctorBase<_, 3, 1>`) on `(high, low, close)`.

*Parameters*:

- `period1` (default `7`): shortest timeframe.
- `period2` (default `14`): medium timeframe.
- `period3` (default `28`): longest timeframe.

*Warmup*: NaN until sample index `max(period1, period2, period3)` (TA-Lib's convention; gates on the longest window).

*Output range*: `[0, 100]`.

## Implementation Details

Composition: tracks `prev_close` as a single scalar plus six `detail::RollingSum` buffers -- one for BP and one for TR at each of the three periods. Each `RollingSum` is O(1) per step, so the total per-step cost is O(1).

## Reference

Bit-exact match to `talib.ULTOSC(high, low, close, timeperiod1, timeperiod2, timeperiod3)` post-warmup (verified to ~1e-14 in `tests/test_third_party_alignment.py`).
