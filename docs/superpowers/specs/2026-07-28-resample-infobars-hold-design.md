# Information bars, OHLC re-aggregation, Hold, and resample/forecast_pairs compliance — design

**Status:** approved for planning (2026-07-28).

**Goal:** Add three operators that surfaced as gaps during BTC/ETH research, and fix
the two design-rule violations the review exposed:
1. **Information bars** (Section 1) — `Resample` cumulative-driver mode
   (volume/dollar bars).
2. **OHLC re-aggregation + `ohlcv` compliance** (Section 2) — `ohlc_bars`/`ohlcv_bars`
   via a C++ multi-column reducer, and retrofit `ohlcv`/`ohlcv2` onto the same
   reducer, deleting the numpy `_resample_ohlcv` shortcut and its eager-only exception.
3. **`Hold`** (Section 3) — a time-based latch functor.
4. **`forecast_pairs` compliance** (Section 4) — reimplement as a `Pipeline` of C++
   nodes so it runs in every regime (streaming a live feature engine into a
   training set).

#1/#2 land on the `Resample` engine; #3 is a new 1->1 functor; #4 is in
`supervised.py`. The compliance parts of #2 and #4 are mandated by the
CONTRIBUTING.md "Design principles" (every operator works in every regime; all logic
in the C++ core), not optional cleanup.

**Conventions (bind all five):** operator logic in C++17, thin Python bindings,
strictly causal, **identical output across eager/graph/lazy regimes** (batch==live,
the definition of done), `nan_policy: ignore`. Each new/changed operator ships: C++
core, binding, a `docs/functions_*/<Name>.md` page with YAML frontmatter (so
`screamer/data/help.json` regenerates via `devtools/build_help_registry.py`), a
runnable `.. plotly::` or `.. exec_code::` example, and tests including an explicit
all-regime/batch==live check and NaN coverage. No version bump is part of this work
(versions move only via `make patch/minor/major` with user approval, after merge).

---

## 1. Information bars — `Resample` cumulative-driver mode

A bar closes when a cumulative *driver* (volume, notional, tick count, anything)
since the last close reaches a `threshold`. This yields volume bars, dollar bars,
and tick bars from one mechanism (Lopez de Prado "information bars").

### Semantics

- New `ResampleMode::ByCumulative` in `include/screamer/dag/resample_params.h`.
- The Resample node gains a **second input, the driver**. Per event it adds the
  driver value to a running accumulator. When `accumulator >= threshold` the
  bucket closes: the crossing observation **is included** in the closing bar, then
  the accumulator resets to `0` (no overshoot carry — the standard convention).
- All existing aggregations (`first/last/min/max/sum/count/mean/ohlc` and the
  `ohlc_bars`/`ohlcv_bars` of section 2), `label` (left/right), `fill`
  (skip/nan/carry), and the arbitrary-functor reducer path apply unchanged: only
  the bucket-boundary rule is new.
- Causal and batch==live: the accumulator is state carried forward; a batch run
  and an event-by-event live run produce identical bars.

### Python API (`screamer/streams.py`)

`threshold=` becomes a fourth **mutually exclusive** mode selector alongside
`freq=`/`every=`/`count=` (presence selects the mode, consistent with the existing
selectors — there is no separate `by=` argument). When `threshold=` is given, a
driver input is **required**:

```python
Resample(price, volume,   threshold=1000, agg='ohlc')   # volume bars
Resample(price, notional, threshold=5e6,  agg='ohlc')   # dollar bars
```

The first input (which may be multi-column) is aggregated; the driver drives the
clock. Tick bars are the existing `count=` and are not a special case here.

### Edge cases

- `threshold <= 0` -> `ValueError` up front (mirrors the `every`/`count` guards).
- A `NaN` driver value is ignored: it does not advance the accumulator and the
  event is handled per the aggregation's own NaN-ignore rule.
- `threshold=` combined with any of `freq/every/count` -> the existing "exactly one
  of" error, extended to name `threshold`.
- Providing `threshold=` without a driver input (or a driver input without
  `threshold=`) -> a clear error.
- The final, unfinished bucket follows the same partial-bucket behavior Resample
  already applies to the other modes.

---

## 2. OHLC re-aggregation — C++ multi-column reducer (+ ohlcv/ohlcv2 retrofit)

Downsample already-built OHLC(V) bars into coarser bars in one node, applying the
correct reducer per column, **entirely in C++**. This is mandated by the design
rules (CONTRIBUTING.md "Design principles"): every operator works in every regime
(eager/graph/lazy) and all logic lives in the C++ core. A Python composition would
violate both — R/WASM bindings would not get the feature — so the numpy-orchestrated
`_resample_ohlcv` shortcut is a pre-existing compliance defect this work removes.

**Approach:** generalize the resample reducer to be multi-column-native in C++. The
`ResampleNode` already receives full-width frames, so a per-column reducer plan runs
in a single pass and works identically in batch, graph, and lazy. Add the column
reducer kinds needed (`First/Max/Min/Last/Sum` already exist as concepts; add
`SumPos`/`SumNeg` so `ohlcv2` stays single-pass). Named aggs map to internal reducer
plans; the public surface stays named strings (no per-column dict is exposed, per
the earlier "no dict-agg" decision).

**Compliance fix (same node):** retrofit `ohlcv`/`ohlcv2` onto the multi-column
reducer, then **delete `_resample_ohlcv` and both eager-only `raise` sites** in
`streams.py`. After this, every resample agg runs in every regime — the one
eager-only inconsistency in the library is gone, not propagated. Backward
compatible: existing batch `ohlcv` calls return the same values; formerly-rejected
graph/lazy calls now succeed.

### Reducer plans (internal, C++)

| agg | input cols | per-column reducers |
|---|---|---|
| `ohlc_bars`  | `[O,H,L,C]` (exactly 4) | first, max, min, last |
| `ohlcv_bars` | `[O,H,L,C,...]` (>= 4)  | first, max, min, last, then **sum on every trailing column** |
| `ohlcv`  (retrofit) | `[price, volume]` (2) | first, max, min, last, sum |
| `ohlcv2` (retrofit) | `[price, signed_volume]` (2) | first, max, min, last, sumPos, sumNeg |

### Two new agg strings

- **`ohlc_bars`**: input columns `[O, H, L, C]` (exactly 4) -> output
  `[first, max, min, last]` = `[O, H, L, C]`.
- **`ohlcv_bars`**: input `[O, H, L, C, ...]` (>= 4) -> OHLC on the first four
  columns; **`sum` on every trailing column**. Covers `[O,H,L,C,V]` and the
  research bar layout `[O,H,L,C,buyV,sellV,buyN,sellN,buyC,sellC]` identically.

```python
Resample(bars, count=5,  agg='ohlcv_bars')   # 1min bars -> 5-bar bars
Resample(bars, freq=300, agg='ohlcv_bars')   # or by a time span
```

These compose with every mode, including the `threshold=` mode of section 1
(volume/dollar OHLCV bars).

### Edge cases

- `ohlc_bars` with != 4 input columns -> `ValueError`.
- `ohlcv_bars` with < 4 input columns -> `ValueError`.
- Output width equals input width; `resample_output_width` / the ohlcv width logic
  is extended so the engine knows the emitted frame width for these aggs.
- Added to `_RESAMPLE_AGGS` and the agg-name validation/error messages.

---

## 3. `Hold` — time-based latch (new 1->1 functor)

Complements `SchmittTrigger` (level hysteresis) with time hysteresis. On a nonzero
finite input it latches that value and shows it for `n` bars total; after `n` bars
with no new trigger it releases to `release` (default `0.0`).

### Semantics (`include/screamer/hold.h`, `ScreamerBase`)

State: `held` value and `remaining` counter.

```
process_scalar(x):
    if isnan(x):                      return NaN           # ignore: state untouched
    if x != 0.0:  held = x; remaining = n - 1;  return x   # (re)trigger
    if remaining > 0:  remaining -= 1;          return held
    return release                                          # expired
```

- `n` = total bars the held value appears (the trigger bar plus `n-1`). `n=1`
  shows the value only on the trigger bar.
- A nonzero input mid-hold re-triggers: replaces `held` and resets `remaining` to
  `n-1`.
- `reset()` sets `remaining = 0` (and `held` to `release`), so before any trigger
  the output is `release`.
- Worked example: `Hold(n=3)([0, 5, 0, 0, 0, -2, 0, 0]) == [0, 5, 5, 5, 0, -2,
  -2, -2]`.

### Constructor / validation

`Hold(n, release=0.0)`: `n` integer `>= 1` else `std::invalid_argument`; `release`
must be finite or `NaN`. O(1) per step, two scalars of state. Bound in
`bindings/bindings_signal.cpp` (alongside `SchmittTrigger`).

---

## 4. `forecast_pairs` — make it all-regime (compliance fix)

`forecast_pairs` (in `screamer/supervised.py`) builds a supervised training set by
delaying features and pairing them with a future target. Today it is eager-only:
it takes arrays, and does the NaN row-dropping (`np.isfinite` masks) and the
duration-mode clock selection (`np.isin`) in numpy. That violates the design rules,
and it is a real user need — someone builds their training set by streaming their
production feature engine, so `forecast_pairs` must consume a live/lazy stream and
emit pairs as they settle.

**Approach:** reimplement `forecast_pairs` as a `Pipeline` of C++ nodes — graph
*structure* built once (allowed), data path entirely in C++:

- count mode: `Input(X) -> Lag(count)`, aligned with `Input(y)`, then `Dropna` when
  `dropna=True`.
- duration mode: `Input(X) -> Delay(duration)`, aligned to y's clock via
  `CombineLatest`, with the y-clock selection expressed by a C++ node (a `Filter`
  on "y ticked", or the appropriate `CombineLatest` emit mode) — not `np.isin` — and
  `Dropna` when requested.

The same `Pipeline` runs in eager, graph, and lazy regimes. No numpy masking or
selection on the data. The public API and return semantics are unchanged
(`(X_shifted, y)`); only the implementation and the accepted input regimes change
(it now also accepts Nodes and lazy event streams). `_leading_nan_mask`,
`_forecast_pairs_duration`, and the numpy masking are removed.

## Testing (all four)

- **Correctness** against hand-derived expected outputs (the worked examples
  above, plus threshold-crossing bars for #1 and per-column reduction for #2).
- **All-regime / batch==live** (definition of done, per CONTRIBUTING.md): the same
  input produces identical output in eager (array), graph (`Pipeline`), and lazy
  (event-iterator) regimes. Covers #1, #2 (including the retrofitted `ohlcv`/`ohlcv2`,
  which previously raised in graph/lazy), #3, and #4.
- **NaN policy**: NaN driver (does not advance the #1 clock), NaN in aggregated
  columns, NaN input to `Hold` (skipped, state untouched).
- **Edge/error**: the rejections listed per section; `n=1` and re-trigger for
  `Hold`; final partial bucket for #1; wrong column count for #2.
- **#4 regressions**: `forecast_pairs` returns the same `(X_shifted, y)` as today for
  batch inputs (count and duration modes, with and without `dropna`), and now also
  runs on Node and lazy stream inputs with identical results.

## Files touched

- C++: `include/screamer/dag/resample_params.h` (+`ByCumulative` mode; a per-column
  reducer plan + `SumPos`/`SumNeg` reducer kinds for the multi-column path),
  `include/screamer/dag/resample_node.h` / `resample_generic_node.h` (driver input +
  cumulative boundary for #1; multi-column reducer for #2/#4), `include/screamer/hold.h`
  (new). Two node-level changes carry the risk and must be detailed in the plan: the
  driver making the node 2-input in `ByCumulative` mode (#1), and the accumulator
  becoming multi-column with a per-column reducer plan (#2/#4).
- Bindings: `bindings/bindings_streams.cpp` (driver input plumbing if needed),
  `bindings/bindings_signal.cpp` (`Hold`).
- Python: `screamer/streams.py` (`threshold=` mode + driver input; `ohlc_bars`/
  `ohlcv_bars` agg names + validation; **route `ohlcv`/`ohlcv2` through the C++
  reducer and delete `_resample_ohlcv` + the two eager-only `raise` sites**).
  `screamer/supervised.py` (reimplement `forecast_pairs` as a `Pipeline`; delete the
  numpy masking helpers).
- Docs: extend `docs/functions_streams/Resample.md` (threshold mode + bar aggs; note
  `ohlcv`/`ohlcv2` now work in all regimes); new `docs/functions_signal/Hold.md`.
  Regenerate `screamer/data/help.json`.
- Tests: extend the resample test module (incl. `ohlcv`/`ohlcv2` now graph/lazy);
  new `tests/test_hold.py`; extend the `forecast_pairs` tests with graph/lazy cases.

## Scope guards (YAGNI)

- #1: reset-to-0 on close, no overshoot carry.
- #2: fixed `ohlc_bars`/`ohlcv_bars` layouts only; no arbitrary per-column agg dict
  (that surface was deliberately removed earlier). The reducer plan is internal.
- #3: nonzero-is-trigger only; no configurable trigger predicate.
- #4: same public API and return values; only the implementation and the accepted
  input regimes change. No new features on `forecast_pairs`.
