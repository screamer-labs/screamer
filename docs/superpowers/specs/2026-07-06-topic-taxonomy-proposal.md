# screamer topic taxonomy (proposal)

A single topic-based `FUNCTIONS` index. Every function carries at least one topic
in its frontmatter; the index page is generated from those declarations, and each
function has one canonical page linked from every topic it belongs to. Topics are
many-to-many by design: a function appears under every lens that fits.

## The topics

| # | Topic | What it covers | count |
|---|---|---|---|
| 1 | Arithmetic | Elementwise numeric ops: rounding, powers, logs, exp, affine, sign, special functions | 17 |
| 2 | Trigonometry & geometry | Angles, inverse trig, coordinate conversions, distances | 10 |
| 3 | Activations | Neural-net activation functions | 6 |
| 4 | Smoothing & moving averages | Track the level of a series, suppress short-term noise | 12 |
| 5 | Filtering | Designed / frequency-domain and state-space filters | 8 |
| 6 | Statistics | Windowed summary statistics: central tendency, dispersion, quantiles, moments, extremes | 26 |
| 7 | Volatility | Size of fluctuations and risk: std/var, RMS, range-based OHLC estimators, ATR | 25 |
| 8 | Standardization & normalization | Put values on a comparable scale: z-scores, affine rescaling, detrending | 4 |
| 9 | Returns & changes | Differences, returns, rates of change, lags | 9 |
| 10 | Cumulative | Running aggregates from t=0 | 8 |
| 11 | Trend | Direction and trend-following: trend MAs, slope fits, ADX, channels | 14 |
| 12 | Momentum & oscillators | Rate-of-change and overbought/oversold indicators | 14 |
| 13 | Bands & channels | Envelopes around price | 3 |
| 14 | Volume | Volume-based indicators | 5 |
| 15 | Correlation & regression | Relationships between series, hedging, pairs, mean-reversion | 14 |
| 16 | Risk & performance | Drawdowns and performance ratios | 8 |
| 17 | Missing data | NaN filling and removal | 4 |
| 18 | Outliers & robustness | Despiking and robust estimators | 7 |
| 19 | Streams | Aligning, combining, reshaping, and replaying event streams | 15 |
| 20 | Computation graphs | Building and running a DAG | 3 |

## Assignments by topic

### 1. Arithmetic
Abs, Ceil, Floor, Round, Clip, Sign, Cube, Square, Sqrt, Power, Exp, Log, Linear, Linear2, Identity, Erf, Erfc

### 2. Trigonometry & geometry
Cos, Sin, Acos, Asin, Atan, Atan2, Hypot, Cart2Polar, Polar2Cart, Tanh

### 3. Activations
Relu, Elu, Selu, Sigmoid, Softsign, Tanh

### 4. Smoothing & moving averages
RollingMean, EwMean, WMA, TRIMA, DEMA, TEMA, KAMA, HullMA, MovingAverage, RollingPoly2, Butter, KalmanFilter

### 5. Filtering
Butter, ButterBandpass, ButterBandstop, ButterHighpass, KalmanFilter, MovingAverage, SchmittTrigger, Hampel

### 6. Statistics
RollingMean, RollingMedian, RollingSum, RollingStd, RollingVar, RollingRms, RollingSkew, RollingKurt, RollingIqr, RollingMad, RollingMedianAD, RollingQuantile, RollingPercentile, RollingRank, RollingMin, RollingMax, RollingMinMax, RollingRange, RollingArgmax, RollingArgmin, EwMean, EwStd, EwVar, EwSkew, EwKurt, EwRms

### 7. Volatility
RollingStd, RollingVar, RollingRms, RollingRange, EwStd, EwVar, EwRms, TrueRange, ATR, NATR, BollingerBands, KeltnerChannels, RollingGarmanKlassVar, RollingGarmanKlassVol, RollingParkinsonVar, RollingParkinsonVol, RollingRogersSatchellVar, RollingRogersSatchellVol, RollingYangZhangVar, RollingYangZhangVol, EwGarmanKlassVar, EwGarmanKlassVol, EwParkinsonVar, EwParkinsonVol, EwRogersSatchellVar, EwRogersSatchellVol

### 8. Standardization & normalization
RollingZscore, EwZscore, Detrend, Linear

### 9. Returns & changes
Diff, Diff2, Lag, Return, LogReturn, Momentum, ROC, ROCP, ROCR

### 10. Cumulative
CumSum, CumProd, CumMax, CumMin, Drawdown, MaxDrawdown, OBV, AD

### 11. Trend
ADX, DonchianChannels, DEMA, TEMA, HullMA, KAMA, MACD, TRIX, RollingPoly1, RollingPoly2, RollingTSF, RollingLinearRegression, RollingHurst, Detrend

### 12. Momentum & oscillators
ADOSC, ADX, BOP, CCI, MACD, MFI, Momentum, ROC, RollingRSI, Stoch, StochRSI, TRIX, UltimateOscillator, WilliamsR

### 13. Bands & channels
BollingerBands, KeltnerChannels, DonchianChannels

### 14. Volume
OBV, MFI, RollingVWAP, AD, ADOSC

### 15. Correlation & regression
RollingCorr, RollingCov, EwCorr, EwCov, EwBeta, RollingBeta, RollingAlpha, RollingLinearRegression, RollingResidualStd, RollingSpread, RollingTSF, RollingHurst, RollingOU, RollingPoly1

### 16. Risk & performance
RollingSharpe, RollingSortino, RollingCalmar, RollingInfoRatio, RollingHitRate, Drawdown, MaxDrawdown, RollingMaxDrawdown

### 17. Missing data
Ffill, FillNa, dropna, dropna_iter

### 18. Outliers & robustness
Hampel, ImpulseClip, RollingSigmaClip, Clip, RollingMedian, RollingMad, RollingMedianAD

### 19. Streams
Stream, merge, merge_iter, combine_latest, combine_latest_iter, replay, dropna, dropna_iter, filter, filter_iter, select, select_iter, split, resample, resample_iter

### 20. Computation graphs
Input, Dag, Node

## Intended overlaps (not errors)

These functions sit in more than one topic on purpose, because users search for
them under different mental models:

- std / var / rms: **Statistics** and **Volatility**
- Butter, KalmanFilter, MovingAverage: **Smoothing** and **Filtering**
- dropna / dropna_iter: **Streams** and **Missing data**
- Clip: **Arithmetic** and **Outliers & robustness**
- RollingMedian, RollingMad, RollingMedianAD: **Statistics** and **Outliers & robustness**
- RollingPoly1: **Trend** and **Correlation & regression**
- RollingPoly2: **Smoothing** and **Trend**
- Momentum, ROC: **Returns & changes** and **Momentum & oscillators**
- MACD, TRIX: **Trend** and **Momentum & oscillators**
- ADX: **Trend** and **Momentum & oscillators**
- MFI, ADOSC, OBV, AD: **Volume** and (Momentum / Cumulative)
- Detrend: **Standardization & normalization** and **Trend**
- Tanh: **Activations** and **Trigonometry & geometry**
- BollingerBands, KeltnerChannels: **Bands & channels** and **Volatility**
- Drawdown, MaxDrawdown: **Risk & performance** and (Cumulative)
- RollingHurst, RollingTSF, RollingLinearRegression: **Trend** and **Correlation & regression**

## Open decisions for you

1. **"Statistics" is the largest bucket (~26).** Keep it as one broad, search-friendly
   topic, or split into Dispersion / Order-statistics / Moments? I lean keep-as-one:
   people do search the word "statistics," and the finance-specific topics (Volatility,
   Risk, Momentum) already give sharper entry points into the same functions.

2. **"Streams" bundles 15 operators.** Keep as one topic, or split into
   "Combining & aligning" (Stream, merge, combine_latest, replay) vs "Reshaping"
   (dropna, filter, select, resample, split)? The notebooks already split this way.

3. **The `_iter` variants** (merge_iter, dropna_iter, ...) currently inherit their
   sibling's topics. Do they get their own index entries, or should they be hidden
   from the topic index and documented on the parent page? I lean hide-from-index to
   cut clutter, since they are the streaming form of the same operator.

4. **"Standardization & normalization" is small (4).** Keep it (z-score is a real,
   distinct search), or fold RollingZscore/EwZscore into Statistics and Linear/Detrend
   elsewhere? I lean keep.

5. **Canonical topic names.** These are the labels users will see and the exact
   strings that go in frontmatter. Lock the spelling/casing now (e.g. "Momentum &
   oscillators") so the frontmatter values and the generated index never drift.
