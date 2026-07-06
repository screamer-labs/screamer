# Despiking filters for screamer — design

**Date:** 2026-07-06
**Status:** approved design, pending spec review
**Topic:** built-in robust despiking (outlier removal) functors

## Goal

screamer has `RollingSigmaClip` (mean/std clipping) but no *robust* despiker. Add
robust, causal, streaming outlier removal as first-class functors, plus the
median-based scale primitive they need. Motivated by a concrete failure: on a
slowly oscillating signal, `RollingSigmaClip`'s window std is inflated by the
signal's own swing, so its band is too wide to catch spikes.

## Deliverables

Three new functors, in the `rolling` family:

1. `RollingMedianAD` — robust scale primitive (median absolute deviation).
2. `Hampel` — the canonical median+MAD despiker.
3. `ImpulseClip` — a causal impulse/glitch remover for non-stationary signals.

## Decisions made during brainstorming

- **Detection scheme:** layered — a canonical raw-domain Hampel as the default,
  and a difference-based detector for non-stationary signals. Measured on the
  motivating signal: canonical raw Hampel caught 9/14 spikes with 16 false
  flags; a naive "raw detection + trend-free scale" was worse (81 false flags and
  it destroyed a genuine level shift); difference-domain detection caught 14/14
  with ~0 false positives. So the non-stationary path detects on the difference.
- **Packaging:** two separate functors (`Hampel`, `ImpulseClip`), not one functor
  with a mode flag. Each name means exactly one algorithm; cleaner docs and
  discoverability. Both reuse the shared `RollingMedianAD` primitive.
- **Causality:** strictly causal, zero latency (respects the project's causality
  hard-rule; batch == stream exact). Consequence for `ImpulseClip`: an impulse is
  a `+/-` doublet in the first difference, so detecting it without lookahead flags
  both the spike sample and its return sample (two replacements per spike, the
  second nudged to the median), and a genuine step's onset loses one sample.
  Single-replacement + step-preservation would require a 1-sample lookahead, which
  we reject. This is why the two-consecutive-replacement behavior is unavoidable,
  not a bug.

## Shared conventions (match existing rolling functors)

- Subclass `ScreamerBase`; implement `process_scalar(double)` and `reset()`. Batch,
  scalar, list, iterator/stream, multi-column, and DAG-node polymorphism come for
  free from `ScreamerBase::operator()`.
- **Causal / batch == stream:** guaranteed by the one-in-one-out `process_scalar`
  model with trailing state only. Verified by the existing
  `tests/test_stream_vs_batch.py` and `tests/test_stream_vs_generator.py` sweeps.
- **NaN policy "ignore":** on a NaN input, leave state untouched and emit NaN
  (as `RollingMedian` / `RollingMad` do).
- **Warmup via `start_policy`** (`"strict"` default): under strict, emit NaN until
  the window is full, like `RollingSigmaClip` and `RollingMad`. Parse with
  `detail::parse_start_policy` into `detail::StartPolicy`.
- Constructor validates `window_size > 0` and the `output` range, throwing
  `std::invalid_argument`, matching `RollingSigmaClip`.

## Core reusable elements (and one refactor)

Existing pieces to reuse: `FixedSizeBuffer` (`common/buffer.h`), `isnan2`
(`common/float_info.h`), `detail::StartPolicy` / `parse_start_policy`
(`detail/start_policy.h`).

**Refactor for reuse:** the streaming-median machinery (two `std::multiset`
halves + `add`/`remove`/`rebalance`/`getMedian`) currently lives inline in
`RollingMedian`. Extract it into a reusable `detail::RollingMedianState` (header
in `include/screamer/detail/`). Then:

- `RollingMedian` uses it (no behavior change; its tests still pass).
- `RollingMedianAD` uses two of them (one for the value median, one for the
  deviation median).
- `Hampel` and `ImpulseClip` use it for the replacement median.

This keeps a single, tested median implementation instead of four copies.

## 1. `RollingMedianAD`

Robust scale: the median absolute deviation over the trailing window,
`MAD_w = median(|x_i - median_w|)`. Unlike the existing `RollingMad` (which is
*mean* absolute deviation, non-robust), this is robust to the outliers we are
trying to detect. `RollingMad` is left unchanged to avoid a breaking rename.

- Signature: `RollingMedianAD(window_size=20, start_policy="strict")`
- Returns the raw median absolute deviation (no 1.4826 scaling; callers scale).
- Implementation: maintain the window; each step compute the window median, then
  the median of the absolute deviations. First version O(W) per step (recompute
  the deviation median from the buffer), matching how `RollingMad` accepts O(W).
  An incremental optimization is a documented follow-up, not a blocker.

## 2. `Hampel`

- Signature: `Hampel(window_size=20, n_sigma=3.0, output=0, start_policy="strict")`
- Detect: `|x[i] - median_w| > n_sigma * 1.4826 * MAD_w` (trailing window only).
- On a flagged sample, emit/record the replacement and feed the **replacement**
  (the median), not the raw outlier, into the window — mirrors `RollingSigmaClip`
  excluding clipped values from its stats, so a burst of spikes cannot pollute the
  scale.
- Best for stationary / mildly varying data and multi-sample outliers.

## 3. `ImpulseClip`

- Signature: `ImpulseClip(window_size=20, n_sigma=4.0, output=0, start_policy="strict")`
- Detect: `|x[i] - x[i-1]| > n_sigma * 1.4826 * MedianAD_w(diff)`, where the scale
  is the robust median-AD of the trailing first differences (trend-free, so it
  works on oscillating signals). Replace flagged samples with `median_w`.
- Higher default `n_sigma` (4.0) than Hampel because the difference domain gives
  cleaner separation.
- Documented costs (measured): the return sample of each spike is also replaced,
  and a real step's onset loses one sample — the price of zero latency.

## Output modes (shared, in the spirit of `RollingSigmaClip`)

- `0` = cleaned signal (outliers replaced by the median) — default
- `1` = outlier flag: `1.0` where flagged, `0.0` otherwise, NaN during warmup
- `2` = input with outliers replaced by NaN (caller fills/interpolates)

Validate `output in {0,1,2}` in the constructor.

## Python bindings

Register in `bindings/bindings_rolling.cpp` following the existing pattern:

```cpp
py::class_<screamer::Hampel, screamer::ScreamerBase>(m, "Hampel")
    .def(py::init<int, double, std::optional<int>, const std::string&>(),
        py::arg("window_size") = 20,
        py::arg("n_sigma") = 3.0,
        py::arg("output") = std::nullopt,
        py::arg("start_policy") = "strict")
    .def("__call__", &screamer::Hampel::operator(), py::arg("value"))
    .def("reset", &screamer::Hampel::reset, "Reset to the initial state.");
```

Analogous blocks for `RollingMedianAD` and `ImpulseClip`. After building, run
`make install-dev` (not just `make build`) so Python imports the fresh binding.

## Docstrings

Each functor gets a class docstring (pybind `py::class_` doc arg or the help.json
entry that feeds the reference pages) covering: what it computes, the detection
rule, the parameters, the output modes, the causal/zero-latency guarantee, and for
`ImpulseClip` the documented two-replacement / step-onset costs. Add reference
pages under `docs/functions_rolling/` and wire them into `topic_rolling.rst`, and
add the metadata that `devtools/build_topic_pages.py` consumes so they appear in
the "Browse by topic" tables.

## Testing

Per-functor test files (`tests/test_rolling_median_ad.py`, `tests/test_hampel.py`,
`tests/test_impulse_clip.py`) following the `test_rolling_mad.py` convention:

- A plain-numpy reference implementation; `assert_allclose(..., equal_nan=True)`
  across several window sizes.
- `RollingMedianAD`: cross-check against `numpy`/`scipy.stats.median_abs_deviation`
  (raw, no scaling) per window.
- `Hampel` / `ImpulseClip`: reference despiker in numpy; assert known spikes are
  removed and known-clean samples are untouched; assert each `output` mode; assert
  behavior on a genuine step (document what each does).
- Edge cases: constant input, all-NaN, single mid-stream NaN, warmup NaNs under
  each `start_policy`, `window_size` validation, `output` validation.

Generic sweeps the new functors must pass (register construction params in
`tests/param_cases.py` so they are picked up automatically):

- `tests/test_stream_vs_batch.py` — causality / batch == stream identity.
- `tests/test_stream_vs_generator.py` — streaming equals generator.
- `tests/test_nan_start_policy_compliance.py` — NaN + start_policy contract.
- `tests/test_view.py` / `test_tensor.py` / `test_matrix.py` — strided and
  multi-column inputs.

## Out of scope / future

- Incremental O(log W) `RollingMedianAD` (start O(W)).
- 1-sample-latency matched-doublet `ImpulseClip` variant (rejected on causality).
- Interpolation-based replacement (non-causal within the gap).
- Wavelet / total-variation despikers (do not fit the streaming causal model).

## Open items for the plan

- Exact placement/name of the extracted `detail::RollingMedianState` header.
- Whether `RollingMedianAD` exposes an optional `scaled` (×1.4826) flag or leaves
  scaling entirely to callers (current lean: leave to callers).
