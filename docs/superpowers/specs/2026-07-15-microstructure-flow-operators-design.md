# screamer: Microstructure & Order-Flow Operators - Design

**Status:** draft design, pending review
**Date:** 2026-07-15
**Scope:** new causal operators in the screamer core library (the algo-trading audience)

## Context

screamer is a causal streaming-operator library (193 operators) whose defining
property is batch == stream: the same operator produces identical results on a
historical array and on a live push/flush feed. An inventory of the current
operators shows the library already covers a lot of quantitative ground, and its
naming families map cleanly onto the two shapes a microstructure model can take:

- `Rolling*` - fixed trailing-window estimators (params `window_size`,
  `start_policy in {strict, expanding, zero}`). The **windowed-parameter** shape.
- `Ew*` - exponentially weighted, recursive (params `com|span|halflife|alpha`).
  The **streaming-prediction / recursive-state** shape.
- `Expanding*` - expanding-window variants.

Already present and reusable (do NOT reinvent): regression/impact primitives
(`RollingBeta`, `RollingLinearRegression`, `EwBeta`, `RollingCov/Corr`,
`ExpandingSlope`, `RollingPoly1/2`); the full realized/range volatility family
(`Ew/Rolling` `GarmanKlass`, `Parkinson`, `RogersSatchell`, `YangZhang`,
`RollingRms`); mean-reversion (`RollingOU`, `RollingHurst`, `Detrend`,
`RollingSpread`); a scalar `KalmanFilter`; volume TA (`OBV`, `AD`, `ADOSC`,
`MFI`, `BOP`); and the structural pieces for multi-stream models (`Pipeline`,
`Input`, `CombineLatest`, `Merge`, `Resample`, `Select`).

The gap is almost entirely **order flow**. There is no trade-sign classifier,
no OFI, no VPIN/PIN, no Hawkes, no propagator, no micro-price, no Amihud, no
Roll spread. This design adds a curated microstructure/order-flow tranche that
fills the gap and, importantly, makes the canonical *named models* discoverable
and educational.

## Goal

A first tranche of causal microstructure operators covering both the
**estimator** side (measure the current flow / impact / liquidity state) and the
**predictor** side (forecast from flow), following screamer's existing
conventions, each canonical model exposed under its well-known name with
teaching-quality documentation.

## Design principles

1. **Fit the existing families.** Windowed estimators are `Rolling*` with
   `window_size` + `start_policy`; recursive predictors are `Ew*` with the
   standard smoothing params. Var/Vol-style paired variants where both make
   sense. Multi-output via the established conventions (fixed-N outputs, e.g.
   `RollingLinearRegression`'s 4, or an `output` selector, e.g. `RollingOU`).

2. **Estimator and predictor, both.** The tranche deliberately spans both
   shapes, and where a model is naturally "estimate parameters, then predict"
   (Kyle's lambda -> predicted impact; a propagator kernel -> convolution
   forecast), it is expressed as a small pipeline of existing + new operators,
   so composition - not a monolith - carries the model.

3. **Named-models layer with documented synonyms.** Popular models get a
   first-class, importable operator under their canonical name. Some are genuinely
   new primitives; others are thin, well-documented **specializations of existing
   general operators** (e.g. `RollingKyleLambda` over `RollingBeta`). Either way,
   the operator's help entry teaches the model. This uses the *existing*
   help.json schema: `title` (canonical name), `short` (one line), `details`
   (formula, intuition, when-to-use, references), `tags`/`topics` (synonyms and
   discoverability, so "price impact", "illiquidity", "Kyle" all find it), and
   `implementation_family` (mark specializations, with a `see_also` to the
   general operator they wrap). No schema change required beyond a `references`
   convention inside `details` and a `see_also` field.

4. **Causality, warmup, parity, discipline.** Every operator is causal (bar t
   uses only data <= t) - screamer's hard rule. Warmup follows `start_policy`
   (honest NaN until filled). batch == stream must hold (the crown-jewel test).
   Operators follow the C++ node-core push/flush/reset contract with thin
   bindings; Python prototypes first, port to C++. Version files are never
   hand-edited (release via `make patch/minor/major`).

## Operator tranche

Kind: **new** = new primitive; **alias** = documented specialization of an
existing operator. Shape: R = Rolling (windowed), E = Ew (recursive),
D = derive/elementwise, P = pipeline of ops.

### A. Trade signing (prerequisite - produces signed flow)
| Operator | Kind | Shape | Inputs -> Outputs | Model |
|---|---|---|---|---|
| `TickRuleSign` | new | D (stateful) | price -> sign {-1,0,+1} | tick rule |
| `LeeReadySign` | new | D (stateful) | price, mid -> sign | Lee-Ready (1991) |
| `BulkVolumeClassifier` | new | R | return, volume, sigma -> buy_frac | BVC, Easley-Lopez de Prado-O'Hara (2012). Signs a *bar's* volume with the return/vol z-score and a normal CDF - works directly on our aggregate bars, no tick data. |

### B. Imbalance
| `OFI` | new | D | buy_vol, sell_vol -> imbalance | Cont-Kukanov-Stoikov (2014), trade-side variant |
| `SignedVolume` | new | D | signed sign, volume -> signed flow | feeds `RollingBeta` for lambda |
| `RollingOrderImbalance` | alias | R | signed flow -> sum/mean | Chordia-Roll-Subrahmanyam; `RollingSum` specialization |

### C. Toxicity / informed trading  (deferred to a later tranche)
| `VPIN` | new | P (Resample + R) | buy_vol, sell_vol on a **volume clock** -> toxicity | Easley-Lopez de Prado-O'Hara (2012). Deferred: needs a volume-clock resampling story we are not building yet. |
| `PIN` | new | R | buy_count, sell_count -> prob. informed | Easley-O'Hara (1996) structural MLE. Deferred: an EM fit, heavier than the rest. |

### D. Price impact / illiquidity
| `RollingKyleLambda` | alias | R | signed flow, return -> lambda (+ R2) | Kyle (1985); specializes `RollingBeta`/`RollingLinearRegression` |
| `EwKyleLambda` | alias | E | signed flow, return -> lambda | recursive variant over `EwBeta` |
| `AmihudIlliquidity` | alias | R/P | return, notional -> illiquidity | Amihud (2002); `\|ret\|/notional` + `RollingMean` |
| `RollSpread` | new | R | price -> effective spread | Roll (1984), from trade autocovariance |

### E. Propagator / long-memory flow -> price  (the centerpiece)
| `Propagator` | new | R (fit) + E/D (predict) | signed flow -> predicted impact | Bouchaud-Gefen-Potters-Wyart (2004): price = sum of a decaying kernel over past signed flow. Estimate the kernel over a window (Shape A), convolve for the prediction (Shape B). Generalizes the "impact is real but decays" finding. |

### F. Self-exciting arrivals (predictor)
| `HawkesIntensity` | new | E | event stream -> intensity | Bacry-Muzy / Bauwens-Hautsch. An exponential-kernel Hawkes is a recursive self-exciting EWMA of events, so it maps directly onto the `Ew*` machinery; captures flow clustering / short-horizon flow momentum. Optional cross-excitation via a 2-input variant. |

### G. Fair value from flow (predictor)
| `MicroPrice` | new | E | bid, ask, imbalance -> fair value | Stoikov (2018); imbalance-adjusted mid, predicts next mid. Needs quote inputs (the caller supplies bid/ask; the bot synthesizes a touch). |

## Documentation approach (the product value)

Each named model's help entry is written to teach:

- `title`: the canonical name (e.g. "Kyle's Lambda").
- `short`: one sentence.
- `details`: the formula, the intuition, when to use it, pitfalls, and
  `references` (author, year, paper) - a few sentences of real content following
  the project writing style (lead with the point, plain language, no em-dashes).
- `tags` / `topics`: synonyms and search terms ("price impact", "illiquidity",
  "market impact", "Kyle") so the operator is found under any of them.
- `see_also`: for aliases, the general operator specialized, plus related models.

A "Microstructure & order flow" section in the docs site groups the tranche and
links the models to each other (signing -> imbalance -> impact -> prediction).

### Example help entry (`RollingKyleLambda`)
```
title: Kyle's Lambda (rolling)
short: Price impact per unit of signed order flow over a trailing window.
details: >
  Kyle (1985) models the price change as linear in signed order flow,
  dp = lambda * flow. lambda is the market-impact / illiquidity coefficient:
  large lambda means a thin book where flow moves price a lot. Estimated here as
  the trailing-window regression slope of return on signed flow; the second
  output is the fit R2 (how much of the move flow explains). lambda is strongly
  time-varying and scales with volatility, so treat it as a liquidity state, not
  a constant. References: Kyle (1985), "Continuous Auctions and Insider Trading".
tags: [price impact, illiquidity, market impact, kyle, lambda, microstructure]
see_also: [RollingBeta, RollingLinearRegression, AmihudIlliquidity]
implementation_family: specialization(RollingBeta)
```

## Scope

**In (first tranche):** signing (`TickRuleSign`, `LeeReadySign`,
`BulkVolumeClassifier`), imbalance (`OFI`, `SignedVolume`,
`RollingOrderImbalance`), impact (`RollingKyleLambda`, `EwKyleLambda`,
`AmihudIlliquidity`, `RollSpread`), propagator (`Propagator`), self-excitation
(`HawkesIntensity`), fair value (`MicroPrice`). Twelve operators, each with a
help entry and tests.

**Language:** the whole first tranche is implemented in **Python** (new module
`screamer/microstructure.py`), conforming to the operator interface (array call
plus push/flush so batch == stream holds). This validates the models and docs
quickly; porting the new primitives to the C++ node core is a separate, later
plan. Aliases are thin Python wrappers over the existing operators.

**Deferred:** `VPIN` (needs a volume-clock resampling story) and `PIN`
(structural EM fit) - a second tranche; L2 order-book models (queue-reactive
Huang-Lehalle-Rosenbaum 2015, Cont-Stoikov-Talreja 2010, book-event OFI) -
require level-2 data screamer users may not have; realized kernels / two-scale
RV (noise-robust vol - a volatility tranche of its own); C++ node-core port of
the tranche's new primitives.

## Correctness and testing

- **Causality:** each operator's value at t is unchanged when future inputs are
  appended (tested by truncation), per the project rule.
- **batch == stream:** every operator produces identical output via a single
  array call and via push/flush live driving (the crown-jewel test).
- **Warmup:** `start_policy` behaves as documented (strict -> NaN until filled).
- **Reference values:** hand-computed checks on scripted inputs (e.g. a known
  linear-impact series recovers lambda; BVC on a known return/sigma matches the
  normal-CDF value; a constructed self-exciting burst raises Hawkes intensity;
  Roll spread on an AR(1)-in-signs price recovers the known spread).
- **Alias equivalence:** each alias equals its underlying general operator on the
  same data (e.g. `RollingKyleLambda == RollingBeta` on (flow, return)).

## Resolved decisions

1. **Alias mechanism:** thin **Python** wrapper that configures the underlying
   operator and carries its own help entry.
2. **`MicroPrice` inputs:** takes explicit bid/ask/imbalance from the caller; the
   core stays data-agnostic.
3. **`PIN`:** deferred to a later tranche (it is an EM fit).
4. **`VPIN` / volume clock:** deferred (no volume-clock resampling in this
   tranche).

## Success criteria

- The tranche's operators install and are discoverable by canonical name and by
  synonym tags.
- Each is causal, passes batch == stream, and has hand-computed reference tests.
- Each named model's docs teach the model (formula, intuition, references).
- A user can compose a full flow pipeline from the pieces - sign -> OFI ->
  `RollingKyleLambda` -> predicted impact - entirely in screamer, running
  identically in batch and live.
