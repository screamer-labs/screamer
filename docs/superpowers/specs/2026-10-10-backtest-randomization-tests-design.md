# Randomization tests for backtest significance

Status: implemented. `position_test` (shift, segments, family-wise for several
strategies), `strategy_test` (bars, returns, shift), the generators
`circular_shift`, `segment_shuffle`, `permute_bars`, `permute_returns`, and the
analytic `probabilistic_sharpe` and `deflated_sharpe`, in Python and JS. The
stationary bootstrap is not implemented.

## Goal

Answer the question "could this backtest result have come from luck?" with a
p-value and a null distribution for any report metric (Sharpe, total PnL, max
drawdown, and so on). The method is to recompute the metric on many randomized
versions of the experiment and see where the real result falls.

## Approaches

### A. Randomize the positions in time

Keep the market data. Replace the strategy's target series with surrogates that
keep its trading profile but break its link to future returns. Replay each
surrogate through the same backtest engine.

Ways to make the surrogates, from most to least faithful:

- **Circular shift.** Rotate the whole target series by a random offset of at
  least `min_shift`. This keeps every property of the series exactly: trade
  count, holding periods, long/short ratio, sizes, time in market, and
  autocorrelation. Only the alignment with prices changes. The wrap-around
  point adds a single artificial transition.
- **Segment shuffle.** Split the target series into runs of constant position
  (trades and flat periods) and shuffle the order of the runs. This keeps the
  exact set of trades, so their durations, sides, and sizes are unchanged. It
  randomizes when they happen more thoroughly than a shift does.
- **Markov surrogate.** Fit a transition matrix on the discretized position
  states and simulate from it. This keeps state occupancy and mean holding
  time, but makes holding times geometric. It is the weakest of the three, and
  the shuffle covers the same need better.

The null hypothesis is: "the timing has no information about future returns,
given this trading profile". It is the classic Monte Carlo permutation test
for a fixed system (Aronson, *Evidence-Based Technical Analysis*, 2006;
Masters, *Permutation and Randomization Tests for Trading System Development*,
2020). It has one important property: a long-biased strategy in a rising market
gets a long-biased null. The test therefore measures timing skill beyond
exposure, which is usually the question that matters.

### B. Randomize the market data

Generate surrogate price paths and re-run the **whole pipeline** on each one:
features, model, and backtest. Common generators:

- **Bar permutation** (Masters). Permute the inter-bar gaps (open over previous
  close) and the intrabar shapes (high, low, and close relative to open)
  separately in log space, then rebuild the path. Every surrogate starts and
  ends at the same prices as the real path and has the same return
  distribution, but no serial structure.
- **Block or stationary bootstrap** of returns (Politis-Romano). This keeps
  short-range dependence such as volatility clustering, so the null is weaker.

Other input columns (volume, funding, external features) are permuted with the
same index, so their relationship to the price within a bar is kept and only
the time order is broken.

This is the gold standard. It is the only approach that also accounts for what
happened inside the pipeline: if the model is fitted or tuned on each
surrogate, the test includes the selection bias from that fitting. The cost is
N full pipeline runs, and with screamer's speed that is usually affordable.

### C. Randomize the model inputs

Keep prices and positions mechanics. Shift or permute one or more feature
inputs so they lose their relationship to future returns, then re-run the
pipeline. This is target shuffling or permutation importance in ML terms. It
answers "does feature X carry information?", which is a question about
attribution, not about whether the strategy is significant. A feature must be
circularly shifted, not permuted bar by bar, so that its autocorrelation is
kept. Otherwise the model sees inputs from a different distribution and its
trading profile changes. When features are derived from the price, C reduces
to B.

## Recommendation

**Build A first, then B. C is a free option of B.**

Why A first:

1. **Common practice.** It is the standard significance test for a given
   position series, and it gives the "timing beyond exposure" null directly.
2. **It does not touch streaming code.** The test needs only the realized target
   series and the market data. The engines already accept `(T, K)` inputs
   (verified: `(T, K)` in, `(T, K, 4)` out). So a full test is: one C++ call
   that builds a `(T, N)` surrogate matrix, one engine call, and one report
   call. All existing streaming nodes stay unchanged, causal, and identical in
   batch and live use. The new code is a separate batch-only module.
3. **It works for every model.** The model is never re-run, so it can be
   anything, including code outside screamer.
4. **It is cheap.** N=1000 replicas cost N engine runs and no model runs.

What A does not cover:

- **Data snooping.** If you tried 50 variants and kept the best, A tests only
  the winner as if it were the only one. B, with the tuning re-run on each
  surrogate, covers this. So does running A on all K variants together and
  using the maximum statistic across them (see below).
- **Quote-posting engines** (`*Orders`). Limit prices are tied to the market
  at that moment, so shifting them in time is meaningless. Use B for these.

Why B second, and C with it: screamer's `Pipeline` already separates the
definition from the data ("define, then bind"). A single driver can bind
surrogate data to the same pipeline N times. Approach B randomizes the market
inputs and approach C randomizes selected feature inputs. Both are one function
with a `randomize=` argument.

## Design

### Layer 1: surrogate generators (C++, batch-only, seeded)

These are new ops in a new topic group, "validation", documented as batch-only.
The rule that all ops are causal applies to operators that compute signals.
These ops generate synthetic data and are never placed inside a live
pipeline. The docs should say so explicitly.

| Op | In | Out | Notes |
|---|---|---|---|
| `CircularShift(n, min_shift, seed)` | `(T,)` | `(T, n)` | Offsets uniform in `[min_shift, T - min_shift]` |
| `SegmentShuffle(n, seed)` | `(T,)` | `(T, n)` | Shuffles runs of constant value; NaN warmup prefix stays in place |
| `PermuteBars(n, seed)` | O, H, L, C [, extra columns] | `(T, n)` per column | Masters' method; extra columns follow the same permutation |
| `StationaryBootstrap(n, mean_block, seed)` | returns `(T,)` | `(T, n)` | Optional; for a null with dependence kept |

The random number generator is a self-contained xoshiro256** generator with
unbiased bounded integers (Lemire's method). Do not use `std::*_distribution`:
its output differs between libstdc++, libc++, and emscripten. With our own
generator, the same seed gives identical surrogates in Python, JS/WASM, and C++,
which keeps the bindings in parity.

### Layer 2: test drivers

```python
res = screamer.validation.position_test(
    BacktestOHLCTarget(taker_fee=5e-4), target, o, h, l, c,
    n=1000, method="shift",          # "shift" | "segments"
    metric="sharpe", min_shift=None, seed=0)
res.observed, res.null, res.p_value   # null: (n,) array

res = screamer.validation.pipeline_test(
    pipeline, data={"o": o, "h": h, "l": l, "c": c},
    randomize="bars",                # B: "bars" | "bootstrap"
    # or randomize={"feature": "shift"}   # C
    n=500, metric="sharpe", seed=0)
```

- **p-value:** `(1 + #{null >= observed}) / (1 + n)` (Phipson and Smyth,
  2010). It is never zero. For metrics where lower is better, the comparison is
  reversed.
- **Several strategies:** if `target` has shape `(T, K)`, the driver returns one
  p-value per strategy, plus a family-wise p-value. The family-wise p-value
  compares the best observed metric with the distribution of the per-replica
  maximum over the K strategies, which is the "max statistic" form of White's
  Reality Check. This addresses data snooping at no extra cost.
- **Memory:** the driver processes the replicas in column chunks, so
  `T = 1e6, n = 1000` never builds the full 8 GB matrix.
- **Where the logic lives:** surrogate generation, the engine runs, and the
  report are all C++ ops. The driver only chains them and counts exceedances,
  like `backtest_report` does today. If that still counts as operator logic in
  Python, a C++ `PermutationPValue` reducer can take over the counting.
  Mirror the driver in `js/src` so the two bindings stay in parity.
- **Default `min_shift`:** about the mean holding period times 2. A smaller
  shift would make some surrogates nearly identical to the real series and
  pull the p-value up.

### Outputs and documentation

- A recipe page, "Is my backtest significant?". It shows a histogram of the null
  distribution with the observed value marked, and the p-value. It runs both
  a skilled signal and a random one, to show what a pass and a fail look
  like.
- Notebook 14 gains a closing section that applies `position_test`.

## Testing

- **Shift and shuffle invariants:** a shift preserves the multiset of values
  and the run lengths, up to the one seam. A segment shuffle preserves the
  multiset of runs exactly.
- **Bar permutation invariants:** same first and last close, same multiset of
  log gaps and intrabar shapes, and `low <= open, close <= high` on every
  surrogate bar.
- **Calibration:** on a random-walk market with a random signal, p-values are
  roughly uniform. A Kolmogorov-Smirnov test over many seeds stays within
  tolerance.
- **Power:** a signal with a planted lead on returns gets a small p-value.
- **Determinism:** a fixed seed gives identical surrogates in Python and JS.
- **Chunking:** the result does not depend on the chunk size.

## Order of work

1. RNG, `CircularShift`, `SegmentShuffle`, `position_test`, the recipe, and
   the JS mirror.
2. `PermuteBars`, `pipeline_test` (approaches B and C), and the family-wise
   p-value.
3. Optional: `StationaryBootstrap` and the deflated Sharpe ratio (López de
   Prado). The deflated Sharpe is an analytic complement that needs no
   resampling.
