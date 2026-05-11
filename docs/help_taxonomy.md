# Help-registry taxonomy

This document defines the **three-layer classification** used by the
help-registry build (`devtools/build_help_registry.py`) and the YAML
frontmatter on each `docs/functions_*/<Name>.md` file.

The three layers serve different jobs and intentionally overlap as
little as possible:

| Layer | Cardinality | Drives | Decided by |
|---|---|---|---|
| `implementation_family` | **single** | Where the sphinx doc page lives (`docs/functions_<family>/`) | Computational pattern |
| `topics` | **multiple** | "Browse by purpose" filters in deep.fund | Use case |
| `tags` | **multiple, free-form** | Keyword search | Specific identifiers / authors |

Sphinx organisation is driven by `implementation_family` only. `topics`
and `tags` are consumed by the help registry (and any future "by topic"
index pages we choose to generate from frontmatter).


## Implementation families (7)

These map 1:1 to the existing `docs/functions_*/` directories. Each
function picks **exactly one**. This is the *computational pattern* of
the function, not its purpose.

| Family | Definition | Examples |
|---|---|---|
| `math` | Stateless element-wise math. No buffer, no recurrence. | `Sin`, `Sqrt`, `Hypot`, `Linear2` |
| `misc` | Stateful but neither rolling-window nor exponentially-weighted. Includes running reductions, lags, returns. | `Lag`, `Return`, `CumSum`, `Detrend`, `Diff` |
| `preprocessing` | Data-cleaning / NaN handling. | `fillna`, `ffill`, `clip` |
| `rolling` | Trailing fixed-size window. | `RollingMean`, `ATR`, `RollingHurst` |
| `ew` | Exponentially-weighted (recurrence-based, no fixed window). | `EwMean`, `EwStd`, `EwCorr` |
| `fin` | Finance-specific accumulators that don't fit `rolling` or `ew` (cumulative, since-inception, event-driven). | `Drawdown`, `MaxDrawdown` |
| `signal` | DSP filters (IIR, FIR, state-space). | `Butter`, `MovingAverage`, `KalmanFilter` |

(Some current placements are debatable — `MACD` and `KAMA` live in
`functions_rolling/` but are more "EW composites". That's
sphinx-history, not a hard constraint. We can shuffle if it matters,
but for this taxonomy we accept the current layout.)


## Topics (17)

These are **use-case** buckets for the deep.fund frontend's "Browse by
purpose" UI. A function picks **all that apply** (typically 1–3).
The list is curated and stable: new topics should only be added when
they have ≥3 members.

### `volatility`
Estimators of dispersion / risk per unit time. Includes range-based,
return-based, and bar-aware variants.

- ✓ `RollingStd`, `EwStd`, `RollingVar`, `EwVar`, `RollingRms`, `EwRms`
- ✓ `RollingParkinson*`, `RollingGarmanKlass*`, `RollingRogersSatchell*`, `RollingYangZhang*`
- ✓ `EwParkinson*`, `EwGarmanKlass*`, `EwRogersSatchell*`
- ✓ `TrueRange`, `ATR`, `NATR`
- ✗ NOT `BollingerBands` (channels), NOT drawdown (risk).

### `volume`
Volume-aware indicators.

- ✓ `OBV`, `AD`, `ADOSC`, `MFI`, `RollingVWAP`

### `trend`
Indicators that estimate the *direction* of a series — slow-moving
averages whose primary use is to identify regimes of monotonic motion.

- ✓ `RollingMean`, `EwMean`, `WMA`, `DEMA`, `TEMA`, `TRIMA`, `HullMA`, `KAMA`, `MACD`
- ✗ NOT generic smoothing filters (those go to `smoothing`).

### `smoothing`
Filters whose *mechanism* is noise reduction. Overlaps `trend` for
most MAs. Distinct from `signal-processing` which also includes
non-smoothing filters (high-pass, band-stop) that *remove* low
frequencies rather than preserve them.

- ✓ `Butter` (low-pass), `MovingAverage`, `KalmanFilter`
- ✓ Also: `EwMean`, `RollingMean` (both `trend` and `smoothing`)
- ✗ NOT `ButterHighpass` / `ButterBandstop` — those are
  `signal-processing` only (they reject smooth components).
- ✗ NOT `KAMA` / `HullMA` etc — those are `trend`-only by convention.

### `momentum`
Rate-of-change family. Outputs proportional to recent change in the
input.

- ✓ `Momentum`, `ROC`, `ROCP`, `ROCR`, `TRIX`, `Diff`, `Diff2`

### `oscillator`
Bounded indicators that swing between fixed extremes (typically 0–100
or −100..+100). Used for overbought/oversold logic.

- ✓ `RollingRSI`, `Stoch`, `StochRSI`, `WilliamsR`, `CCI`, `UltimateOscillator`, `BOP`
- ✗ NOT `MACD` (unbounded — that's `trend`).

### `channels`
Bands / envelopes around price.

- ✓ `BollingerBands`, `DonchianChannels`, `KeltnerChannels`

### `risk`
Backtest-evaluation metrics: drawdowns, risk-adjusted returns, hit
rates. Things you'd put on a tearsheet.

- ✓ `Drawdown`, `MaxDrawdown`, `RollingMaxDrawdown`
- ✓ `RollingSharpe`, `RollingSortino`, `RollingInfoRatio`, `RollingCalmar`, `RollingHitRate`

### `regression`
Fits a parametric model (line, polynomial) to a window.

- ✓ `RollingPoly1`, `RollingPoly2`, `RollingTSF`, `RollingLinearRegression`
- ✓ `RollingBeta`, `RollingAlpha`, `RollingResidualStd`
- ✓ `Linear`, `Linear2` (stateless, but fit-shape)

### `statistics`
Distributional summaries beyond mean/variance — order statistics,
shape statistics, ranks.

- ✓ `RollingSkew`, `RollingKurt`, `EwSkew`, `EwKurt`
- ✓ `RollingMedian`, `RollingQuantile`, `RollingMin`, `RollingMax`, `RollingMinMax`
- ✓ `RollingRange`, `RollingMad`, `RollingIqr`, `RollingArgmin`, `RollingArgmax`
- ✓ `RollingZscore`, `EwZscore`
- ✓ `RollingHurst`, `RollingRank`, `RollingPercentile`
- ✗ NOT `RollingMean`/`EwMean` (those are `trend`/`smoothing`).

### `correlation`
Pair-statistics. Split from `statistics` because "find me a
correlation indicator" is a frequent query.

- ✓ `RollingCorr`, `RollingCov`, `RollingBeta`, `RollingSpread`
- ✓ `EwCov`, `EwCorr`, `EwBeta`

### `transforms`
Element-wise or stateful series transformations whose purpose is to
reshape data for downstream use (delay, detrend, cumulate). Distinct
from `math` (which is pure stateless arithmetic).

- ✓ `Lag`, `Diff`, `Diff2`, `Detrend`
- ✓ `Return`, `LogReturn` (with `returns` tag)
- ✓ `CumSum`, `CumProd`, `CumMax`, `CumMin`, `Identity`

### `math`
Stateless arithmetic / scalar functions.

- ✓ `Abs`, `Sign`, `Sqrt`, `Square`, `Cube`, `Exp`, `Log`, `Erf`, `Erfc`
- ✓ `Floor`, `Ceil`, `Round`, `Power`
- ✓ `Sin`, `Cos`, `Atan`, `Asin`, `Acos`, `Atan2`, `Hypot`
- ✗ NOT activation functions (those are `activation`).

### `geometry`
2D coordinate conversions and related.

- ✓ `Cart2Polar`, `Polar2Cart`
- ✓ Also tagged on: `Hypot`, `Atan2` (which compute the polar components)

### `activation`
Neural-network-style nonlinearities. Useful as composition primitives
for feature engineering.

- ✓ `Relu`, `Elu`, `Selu`, `Sigmoid`, `Tanh`, `Softsign`

### `data-handling`
Clean / sanitize / pre-process input. Outlier handling lives here.

- ✓ `fillna`, `ffill`, `clip`
- ✓ `RollingSigmaClip` (outlier removal)

### `signal-processing`
DSP filters and adaptive estimators. Overlaps `smoothing` heavily;
distinct because users coming from a DSP background look here.

- ✓ `Butter`, `ButterHighpass`, `ButterBandpass`, `ButterBandstop`
- ✓ `MovingAverage` (FIR), `KalmanFilter`


## Tags

Tags are **free-form** keywords for search — not a controlled
vocabulary. Use them for:

- **Identifiers / authors**: `wilder`, `kaufman`, `mulloy`, `peters`, `parkinson`, `garman-klass`, `anis-lloyd`
- **Acronyms / aliases**: `rsi`, `macd`, `obv`, `r/s`, `wma`, `ema`, `sma`
- **Concepts not big enough to be topics**: `long-memory`, `fractal`, `mean-reversion`, `anomaly`, `standardization`, `range-based`, `bar-aware`, `iir`, `fir`
- **Input requirements**: `ohlc`, `hlc`, `hl`, `requires-volume`, `pair` (two inputs)

Rule of thumb: if a concept has ≥3 functions and users would
*browse* by it, promote it to a topic. Otherwise it stays a tag.


## Examples

```yaml
# RollingHurst
implementation_family: rolling
topics: [statistics]
tags: [hurst, long-memory, fractal, r/s, anis-lloyd, regime]
```

```yaml
# RollingSigmaClip
implementation_family: rolling
topics: [data-handling]
tags: [outlier, anomaly, sigma-clip]
```

```yaml
# KAMA
implementation_family: rolling
topics: [trend]
tags: [kaufman, adaptive, ema]
```

```yaml
# RollingParkinsonVol
implementation_family: rolling
topics: [volatility]
tags: [parkinson, range-based, ohlc, hl]
```

```yaml
# OBV
implementation_family: rolling
topics: [volume]
tags: [obv, requires-volume]
```

```yaml
# Atan2
implementation_family: math
topics: [math, geometry]
tags: [trigonometry, polar, pair]
```


## Resolved borderline cases

- **`smoothing` vs `signal-processing`** — kept separate. Signal-
  processing covers filters that *remove* signal content (high-pass,
  band-stop, band-pass), which aren't smoothing.
- **`returns`** — demoted to tag. `Return` and `LogReturn` go in
  `transforms` topic with `returns` tag.
- **`MACD`** — `trend` (despite the name).
- **Borderline functions** — tag generously, topic-assign sparingly.

Every function in `screamer/__init__.py` gets frontmatter following
this taxonomy, and the help registry build validates the result.
