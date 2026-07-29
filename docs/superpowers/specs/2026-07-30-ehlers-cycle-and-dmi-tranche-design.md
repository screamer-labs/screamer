# screamer: Cycle/Spectral (Ehlers) and DMI Operator Tranche - Design

**Status:** draft design, pending review
**Date:** 2026-07-30
**Scope:** new causal operators in the screamer core library, in two tranches

## Context

A coverage comparison against TA-Lib (162 functions across 10 groups) shows
screamer already covers the statistical, volatility, math, and volume ground
well, and ships a large surface TA-Lib lacks (microstructure, backtesting, risk
ratios, designed filters, streaming/pipelines). Two coherent groups are absent:

- **Cycle Indicators (TA-Lib group, 0 of 5 present).** The entire Hilbert
  transform family: dominant cycle period, phase, phasor, sinewave, trend mode.
- **Directional Movement (part of TA-Lib Momentum, ~6 present as one bundle).**
  `ADX` already computes +DI, -DI, DX, and the +DM/-DM building blocks internally
  and is bit-exact to TA-Lib, but only exposes the `(+DI, -DI, ADX)` triple. The
  individual members are not importable.

The two groups differ in one decision that shapes the whole tranche. For the
cycle group, the TA-Lib `HT_*` implementation is a frozen early-1990s port of
John Ehlers' work that runs the Hilbert transform on raw price. Ehlers later
superseded it: raw price carries a trend component and spectral dilation that
contaminate the cycle measurement, and his fix is a roofing filter (a highpass
to remove the trend, then a low-lag lowpass to remove aliasing noise) applied
before any cycle measurement. This tranche implements the later Ehlers
formulation, not the TA-Lib port. For directional movement there is no superseding
variant, so those operators match the standard Wilder definition and baseline
against TA-Lib.

## Goal

Two tranches of causal operators, each member importable under a screamer-style
name with teaching-quality documentation:

1. **Cycle/spectral (Ehlers).** A layer of reusable causal filters, an
   analytic-signal and dominant-cycle core built on those filters, and two
   adaptive overlays. Ehlers' later formulation throughout.
2. **Directional movement (DMI).** The individual Wilder directional-movement
   operators, sharing one implementation with the existing `ADX`.

## Design principles

1. **Modern routine over legacy parity.** Where a classic function is outdated
   and a better variant exists, ship the better variant. The cycle tranche
   follows Ehlers' current formulation rather than a bit-exact TA-Lib port.
   Feature-completeness against TA-Lib is a secondary benefit, not the goal.

2. **Reusable components are first-class operators.** The roofing filter and its
   parts (SuperSmoother, highpass-derived Decycler) are causal filters with value
   on their own. They ship as importable operators, and the cycle core consumes
   them rather than hiding them.

3. **One implementation per behavior.** The DMI math already lives inside `ADX`.
   It moves to a shared header that both `ADX` and the new DMI operators call. No
   second copy of the directional-movement recurrence.

4. **Reuse the existing filter substrate.** `Butter` and `ButterHighpass` are
   coefficient recipes feeding a shared `IIRFilter` (`signal/signal.h`). The
   fixed-period Ehlers filters use the same substrate with Ehlers' coefficient
   formulas. SuperSmoother is a distinct design from Butterworth (lower lag by
   construction), so it is a new recipe, not a duplicate.

5. **Every operator in every regime, causal, batch == stream.** Each operator
   runs eager, graph, and lazy with identical output, depends only on current and
   past input, and passes the stream-vs-batch tests. Warm-up emits `NaN` and is
   identical across regimes.

## Naming

Names follow screamer conventions, not TA-Lib's. Established short acronyms stay
bare (as `ADX`, `KAMA` already do); everything else is descriptive PascalCase (as
`SuperSmoother`, `DonchianChannels`, `UltimateOscillator` already are). TA-Lib's
`HT_*`, `PLUS_DI` forms are not used.

| Layer | screamer name | Outputs | TA-Lib equivalent |
|---|---|---|---|
| Filters | `SuperSmoother` | 1 | (none, Ehlers) |
| Filters | `RoofingFilter` | 1 | (none, Ehlers) |
| Filters | `Decycler` | 1 | (none, Ehlers) |
| Cycle core | `DominantCycle` | 1 | `HT_DCPERIOD` |
| Cycle core | `CyclePhase` | 1 | `HT_DCPHASE` |
| Cycle core | `CycleAmplitude` | 1 | (envelope; implicit in TA) |
| Cycle core | `HilbertPhasor` | 2 (inphase, quadrature) | `HT_PHASOR` |
| Cycle core | `CycleSine` | 2 (sine, leadsine) | `HT_SINE` |
| Cycle core | `TrendMode` | 1 | `HT_TRENDMODE` |
| Adaptive | `InstantaneousTrendline` | 1 | `HT_TRENDLINE` |
| Adaptive | `MAMA` | 2 (mama, fama) | `MAMA` |
| DMI | `PlusDI`, `MinusDI` | 1 each | `PLUS_DI`, `MINUS_DI` |
| DMI | `PlusDM`, `MinusDM` | 1 each | `PLUS_DM`, `MINUS_DM` |
| DMI | `DX`, `ADXR` | 1 each | `DX`, `ADXR` |

## Tranche 1: cycle/spectral (Ehlers)

Four layers, each built on the one below.

### Layer 1: reusable causal filters

Each is a `ScreamerBase` node parameterized by a period, computing Ehlers'
coefficients and feeding the shared `IIRFilter`, in the same shape as `Butter`.

- **`SuperSmoother(period)`.** Ehlers' 2-pole low-lag lowpass. Coefficients from
  the standard `exp(-sqrt(2) pi / period)` / `cos(sqrt(2) pi / period)` form.
- **`Decycler(period)`.** Trend estimate: input minus its highpass component.
  Removes cycles at or below `period`, leaving the trend.
- **`RoofingFilter(hp_period, lp_period)`.** Highpass at `hp_period` then
  `SuperSmoother` at `lp_period`. A bandpass that isolates the tradeable cycle
  band. Composes the highpass and SuperSmoother inside one C++ node, as `MACD`
  composes `EwMean`. This is the preprocessing every cycle-core operator applies.

The Ehlers highpass that `Decycler` and `RoofingFilter` share is not a standalone
operator; its coefficient recipe lives in a `detail/` header both call, so there
is one implementation of it. `ButterHighpass` stays the public highpass (a
different, Butterworth design).

`nan_policy: ignore`. Docs family `signal`, topic `filtering`.

### Layer 2: analytic signal and dominant cycle

Each operator applies the roofing filter to the input, forms the analytic signal
by Ehlers' quadrature method, and reads the requested quantity from it. The
dominant cycle uses the homodyne discriminator (Ehlers' current estimator). The
period feeds back into the measurement from past bars only, so causality and
batch == stream hold; the settling interval is a `NaN` warm-up.

- **`HilbertPhasor(...)`** returns `(inphase, quadrature)`, the real and imaginary
  parts of the analytic signal. The building block the others read from.
- **`DominantCycle(...)`** returns the measured cycle period in samples.
- **`CyclePhase(...)`** returns the instantaneous phase in degrees, 0 to 360.
- **`CycleAmplitude(...)`** returns the instantaneous amplitude (cycle envelope).
- **`CycleSine(...)`** returns `(sine, leadsine)`, the sinewave-indicator pair.
- **`TrendMode(...)`** returns a trend-versus-cycle classification.

`nan_policy: ignore`. Docs family `signal`, new topic `cycles`.

Shared roofing/analytic-signal machinery lives in a `detail/` header so the six
operators read from one implementation rather than each re-deriving the quadrature.

### Layer 3: adaptive overlays

- **`InstantaneousTrendline(...)`.** Ehlers' causal IIR trendline whose smoothing
  tracks the measured dominant cycle. Docs family `signal`, topic `smoothing`.
- **`MAMA(...)`** returns `(mama, fama)`, the MESA Adaptive Moving Average and its
  following average. The adaptation rate is driven by the rate of change of the
  measured phase. The acronym matches the `KAMA`/`DEMA`/`TEMA` convention. Docs
  family `signal` or `rolling` (alongside the moving averages), topic `smoothing`.

## Tranche 2: directional movement (DMI)

`ADX`'s per-bar TR, +DM, -DM, the Wilder smoothing recurrence, and the +DI/-DI/DX
computation move into `include/screamer/detail/dmi_core.h`. `ADX` is refactored to
call it with no change to its output (verified against its existing baseline). New
thin operators call the same core:

- **`PlusDI(window_size=14)`, `MinusDI(window_size=14)`** return the smoothed
  directional indicators.
- **`PlusDM(window_size=14)`, `MinusDM(window_size=14)`** return the smoothed
  directional movement.
- **`DX(window_size=14)`** returns the directional index (the pre-average input to
  `ADX`).
- **`ADXR(window_size=14)`** returns the average directional index rating, the mean
  of `ADX` now and `ADX` `window_size` bars ago.

All take `(high, low, close)` where `ADX` does (`ADXR` and `DX` need the same three;
`PlusDM`/`MinusDM` need `high`, `low`). `nan_policy: ignore`, matching `ADX`. Docs
family `fin`, topics `trend`/`momentum`. Baseline against TA-Lib (`PLUS_DI` etc.),
which is bit-exact-matchable here.

## Definition of done (per operator, from CONTRIBUTING.md)

1. C++ functor plus thin pybind11 binding: filters and cycle core in
   `bindings_signal.cpp`; DMI in `bindings_fin.cpp` (alongside `ADX`).
2. Runs eager, graph (`Pipeline`), and lazy with identical output; a test proves
   batch == live for each operator. This is the definition of done.
3. Polymorphic argument handling; causal; passes stream-vs-batch tests.
4. `docs/functions_<family>/<Name>.md` with validated frontmatter, including a
   declared `nan_policy` and topics from `docs/topics.yml`.
5. One new topic slug `cycles` added to `topics.yml` under the `indicators` group
   (definition: "Dominant cycle, phase, amplitude, and trend/cycle mode from the
   analytic signal."). Filters reuse `filtering`; MAMA/trendline reuse `smoothing`.
6. Baselines in `devtools/baselines/`: TA-Lib for the DMI family; a reference
   Python implementation of Ehlers' published equations for the filters and cycle
   operators (no bit-exact library oracle exists for the modern formulation, by
   choice).
7. `make tidy` clean; `make build` regenerates `screamer/__init__.py`.

## Out of scope

- The other missing TA-Lib Momentum members (`AROON`, `AROONOSC`, `PPO`, `APO`,
  `CMO`). Independent standalone operators, a separate follow-up tranche.
- Candlestick pattern recognition (61 `CDL*` functions). A deliberate scope
  decision to revisit separately, not an oversight.
- Parabolic SAR, T3, MAVP, and the trivial price-transform/math one-liners
  (`AVGPRICE`, `LOG10`, and so on). Parity-only follow-ups.
- A standalone Ehlers bandpass and instantaneous-frequency operator. Their main
  use is internal to the cycle core; expose later if a user needs them directly.

## Open questions

1. `MAMA` docs family and topic: group it with the moving averages (`rolling`,
   `smoothing`) or with the cycle operators (`signal`, `cycles`)? It is a moving
   average whose mechanism is the cycle core.
2. `CycleAmplitude` has no distinct TA-Lib counterpart. Confirm it earns a place
   as a first-class operator rather than being left implicit.
3. Ehlers' filters are parameterized by `period` (samples), whereas `Butter` uses
   a normalized cutoff. Confirm `period` is the friendlier argument for this
   family and that the two conventions coexisting is acceptable.
