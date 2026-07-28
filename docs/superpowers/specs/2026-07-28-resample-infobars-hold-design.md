# Information bars, OHLC re-aggregation, and Hold — design

**Status:** approved for planning (2026-07-28).

**Goal:** Add three operators to screamer that surfaced as gaps during BTC/ETH
research: (1) information bars (event-clock resampling on a cumulative driver),
(2) OHLC bar re-aggregation, and (3) `Hold`, a time-based latch. #1 and #2 extend
the existing `Resample` engine; #3 is a new 1->1 functor.

**Conventions (bind all three):** operator logic in C++17, thin Python bindings,
strictly causal, batch output identical to live/streaming output, `nan_policy:
ignore`. Each ships: C++ core, binding, a `docs/functions_*/<Name>.md` page with
YAML frontmatter (so `screamer/data/help.json` regenerates via
`devtools/build_help_registry.py`), a runnable `.. plotly::` or `.. exec_code::`
example, and tests including an explicit batch==live check and NaN coverage. No
version bump is part of this work (versions move only via `make patch/minor/major`
with user approval, after merge).

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

## 2. OHLC re-aggregation — fixed-layout aggs

Downsample already-built OHLC(V) bars into coarser bars in one node, applying the
correct reducer per column. All per-column compute stays in C++, and the result
must be identical in batch and in the graph/live regime (batch==live). The current
`ohlcv` path composes C++ reducers via Python array-splitting, which is a batch
orchestration; since these bar aggs must also hold batch==live through the
Node/`Pipeline` path, the implementation plan chooses between: (a) a dedicated C++
columnar OHLC-bars reducer — preferred, one node, uniform batch and live; or (b)
extending the existing `ohlcv`-style composition only if it already supports the
graph regime. Either way Python does no numeric work.

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

## Testing (all three)

- **Correctness** against hand-derived expected outputs (the worked examples
  above, plus threshold-crossing bars for #1 and per-column reduction for #2).
- **batch==live**: the same input as a single array and as an event-by-event
  stream produces identical output (the crown-jewel invariant). For #1/#2 via the
  `Pipeline`/Node path; for #3 via the scalar-loop vs array path.
- **NaN policy**: NaN driver (does not advance the #1 clock), NaN in aggregated
  columns, NaN input to `Hold` (skipped, state untouched).
- **Edge/error**: the rejections listed per section; `n=1` and re-trigger for
  `Hold`; final partial bucket for #1; wrong column count for #2.

## Files touched

- C++: `include/screamer/dag/resample_params.h` (+`ByCumulative` mode and its
  boundary handling; a C++ OHLC-bars reducer if the plan picks option (a) for #2),
  `include/screamer/dag/resample_node.h` / `resample_generic_node.h` (driver input,
  cumulative boundary), `include/screamer/hold.h` (new). The driver turns the
  Resample node into a 2-input node in `ByCumulative` mode — the main implementation
  risk, to be detailed in the plan.
- Bindings: `bindings/bindings_streams.cpp` (driver input plumbing if needed),
  `bindings/bindings_signal.cpp` (`Hold`).
- Python: `screamer/streams.py` (`threshold=` mode, driver input, `ohlc_bars`/
  `ohlcv_bars` aggs + validation).
- Docs: extend `docs/functions_streams/Resample.md` (threshold mode + bar aggs);
  new `docs/functions_signal/Hold.md`. Regenerate `screamer/data/help.json`.
- Tests: extend the resample test module; new `tests/test_hold.py`.

## Scope guards (YAGNI)

- #1: reset-to-0 on close, no overshoot carry.
- #2: fixed `ohlc_bars`/`ohlcv_bars` layouts only; no arbitrary per-column agg dict
  (that surface was deliberately removed earlier).
- #3: nonzero-is-trigger only; no configurable trigger predicate.
