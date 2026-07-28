# Information bars, OHLC re-aggregation, Hold, and compliance — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Add three operators (information bars, OHLC re-aggregation, `Hold`) and fix
the two design-rule violations they exposed (`ohlcv`/`ohlcv2` eager-only numpy
shortcut; `forecast_pairs` eager-only), all C++-core and all-regime.

**Architecture:** #1/#2 extend the C++ `Resample` node (`include/screamer/dag/`);
#3 is a new `ScreamerBase` functor; #4 reimplements `forecast_pairs` as a `Pipeline`
of existing C++ nodes. Spec: `docs/superpowers/specs/2026-07-28-resample-infobars-hold-design.md`.

**Tech Stack:** C++17 + pybind11 (scikit-build-core), Python 3.11+, pytest.

## Global Constraints (from CONTRIBUTING.md "Design principles" — every task)

- **All operator logic in the C++ core.** No numeric compute, no chaining/gluing of
  op outputs, no data-path orchestration in Python. The only sanctioned combination
  is a `Pipeline`/DAG (graph structure built once, data path in C++).
- **Every operator works in every regime** — eager (arrays), graph (`Pipeline`/Node),
  lazy (event iterator) — with **identical output** (batch==live). No eager-only ops.
  An all-regime/batch==live test is the definition of done.
- `nan_policy: ignore` for the new operators. Causal: output at t uses only inputs <= t.
- After any C++ change: `make install-dev` before testing. Run the full suite before
  marking a task done. **No version-file edits** (versions move only via `make
  patch/minor/major`, after merge).
- Each new/changed public operator: C++ core + thin binding + `docs/functions_*/<Name>.md`
  with YAML frontmatter + regenerate `screamer/data/help.json` (`poetry run python
  devtools/build_help_registry.py`) + tests.
- Commit after each green task. Do NOT bump the version.

## Two required test helpers (write once, in Task 1, reuse everywhere)

Add to a shared test module `tests/regime_helpers.py`:

```python
import numpy as np

def as_batch(op_factory, *arrays):
    """Run an op on concrete arrays (eager regime). op_factory() -> a fresh op."""
    return np.asarray(op_factory()(*arrays))

def as_scalar_loop(op_factory, *arrays):
    """Feed the op one row at a time (the functor scalar-loop / lazy analogue)."""
    op = op_factory()
    n = len(arrays[0])
    out = [op(*[np.asarray(a, float)[i] for a in arrays]) for i in range(n)]
    return np.asarray(out, float)

def assert_batch_equals_scalar(op_factory, *arrays):
    """Crown-jewel: array call == row-by-row call, NaN-aware."""
    b = as_batch(op_factory, *arrays)
    s = as_scalar_loop(op_factory, *arrays)
    np.testing.assert_allclose(b, s, equal_nan=True)
```

For stream operators (Resample, forecast_pairs) the "live" regime is the
`Pipeline` lazy path; each task below gives the exact graph/lazy assertion.

---

## Task 1: `Hold` functor (new 1->1 operator)

**Files:**
- Create: `include/screamer/hold.h`
- Modify: `bindings/bindings_signal.cpp` (register `Hold`, next to `SchmittTrigger`)
- Create: `docs/functions_signal/Hold.md`
- Create: `tests/test_hold.py`, `tests/regime_helpers.py`

**Interfaces:**
- Produces: `screamer.Hold(n: int, release: float = 0.0)`, a 1->1 op. On a nonzero
  finite input it latches that value and outputs it for `n` bars total (trigger bar
  + n-1); a nonzero input mid-hold replaces the value and resets the counter; after
  `n` bars with no trigger it outputs `release`. NaN input -> NaN, state untouched.
  `n >= 1` else `std::invalid_argument`; `release` finite or NaN.

- [ ] **Step 1 — failing functional test.** In `tests/test_hold.py`:

```python
import math, numpy as np, pytest
from screamer import Hold
from tests.regime_helpers import assert_batch_equals_scalar

def test_hold_worked_example():
    out = np.asarray(Hold(n=3)(np.array([0.,5,0,0,0,-2,0,0])))
    np.testing.assert_array_equal(out, [0,5,5,5,0,-2,-2,-2])

def test_hold_n1_shows_only_trigger_bar():
    out = np.asarray(Hold(n=1)(np.array([0.,5,0,7,0])))
    np.testing.assert_array_equal(out, [0,5,0,7,0])

def test_hold_release_value():
    out = np.asarray(Hold(n=2, release=-1.0)(np.array([0.,5,0,0,0])))
    np.testing.assert_array_equal(out, [-1,5,5,-1,-1])

def test_hold_nan_ignored_state_untouched():
    out = np.asarray(Hold(n=3)(np.array([0.,5,np.nan,0,0])))
    # NaN passes through; the hold counter does NOT advance on the NaN bar
    assert math.isnan(out[2])
    np.testing.assert_array_equal(out[[0,1,3,4]], [0,5,5,5])

def test_hold_rejects_n_below_1():
    with pytest.raises(ValueError):
        Hold(n=0)

def test_hold_batch_equals_scalar():
    rng = np.random.default_rng(0)
    x = np.where(rng.random(200) < 0.1, rng.standard_normal(200), 0.0)
    x[::17] = np.nan
    assert_batch_equals_scalar(lambda: Hold(n=5), x)
```

- [ ] **Step 2 — run, verify it fails** (`Hold` not defined). `make install-dev` not needed yet.
- [ ] **Step 3 — implement `include/screamer/hold.h`** mirroring `include/screamer/schmitt_trigger.h`:
  `ScreamerBase` subclass; ctor `Hold(int n, double release=0.0)` validates `n>=1`
  (throw `std::invalid_argument`) and `release` finite-or-NaN; state `held_`,
  `remaining_`; `reset()` sets `remaining_=0`, `held_=release_`.
  `process_scalar(x)`: `if isnan2(x) return NaN;` `if (x != 0.0){ held_=x; remaining_=n_-1; return x;}`
  `if (remaining_>0){ --remaining_; return held_;}` `return release_;`
- [ ] **Step 4 — register in `bindings/bindings_signal.cpp`** with the `SchmittTrigger`
  pattern (class_, ctor args `py::arg("n"), py::arg("release")=0.0`, `__call__`).
- [ ] **Step 5 — `make install-dev`; run the tests; all pass.**
- [ ] **Step 6 — `docs/functions_signal/Hold.md`** with frontmatter (`name: Hold`,
  `title`, `implementation_family: signal`, `topics: [signal]` or matching, `tags`,
  `short`, `inputs: 1`, `outputs: 1`, `parameters:` n (int, min 1) and release (float,
  default 0.0), `nan_policy: ignore`), a Description leading with what it computes, and
  a runnable `.. plotly::` example. Regenerate: `poetry run python
  devtools/build_help_registry.py`.
- [ ] **Step 7 — run `python -m pytest -q` (full suite) + `make docs` (clean). Commit.**

---

## Task 2: `forecast_pairs` all-regime (compliance)

**Files:**
- Modify: `screamer/supervised.py` (reimplement as a `Pipeline`; delete numpy masking)
- Test: extend/replace `tests/test_supervised_forecast_pairs.py`

**Interfaces:**
- `forecast_pairs(X, y, *, count=None, duration=None, dropna=False)` -> `(X_shifted, y)`.
  Public API and batch return values UNCHANGED. Now also accepts `Node` inputs (returns
  a Node) and lazy event streams (returns a lazy iterator), producing identical results.
- Consumes existing C++ ops: `Lag`, `Delay`, `CombineLatest`, `Dropna`, `Input`,
  `Pipeline` (all in `screamer/`), which are already all-regime.

**Approach:** build one `Pipeline`. count mode: `Xs = Lag(count)(Input('X'))`, combine
with `Input('y')`, then `Dropna(how='any')` when `dropna`. duration mode: `Delay(duration)`
on X, `CombineLatest` to y's clock, select y-tick rows via a C++ node (not `np.isin`),
`Dropna` when requested. Bind data at call time. Remove `_leading_nan_mask`,
`_forecast_pairs_duration`, and all `np.isfinite`/`np.isin` masking.

- [ ] **Step 1 — regression test first: batch output must not change.** Keep the existing
  batch assertions (count + duration, with/without dropna) — they must still pass byte-for-byte.
- [ ] **Step 2 — add the all-regime tests** (these fail today because it's eager-only):

```python
import numpy as np
from screamer import forecast_pairs, Input, Pipeline

def test_forecast_pairs_count_graph_matches_batch():
    X = np.arange(10.0); y = X * 2
    Xb, yb = forecast_pairs(X, y, count=2)
    # graph regime: build once, bind data
    node = forecast_pairs(Input('X'), Input('y'), count=2)   # returns a Node
    # (exact run API per screamer.Pipeline; assert identical to (Xb, yb))
    ...

def test_forecast_pairs_count_lazy_matches_batch():
    X = np.arange(10.0); y = X * 2
    Xb, yb = forecast_pairs(X, y, count=2)
    events_X = iter(list(enumerate(X))); events_y = iter(list(enumerate(y)))
    # lazy regime: iterator in -> iterator out; collect and compare to (Xb, yb)
    ...
```

(The implementer fills the exact graph/lazy run calls by matching how other stream
ops — `resample`, `dropna` — are exercised in `tests/test_streams_*.py`.)

- [ ] **Step 3 — compliance test: no numpy data masking remains.**

```python
import inspect, screamer.supervised as S
def test_no_numpy_datapath_in_supervised():
    src = inspect.getsource(S)
    for banned in ("np.isfinite", "np.isin", "[keep]", "[m]", "_leading_nan_mask"):
        assert banned not in src, f"data-path numpy still present: {banned}"
```

- [ ] **Step 4 — implement the `Pipeline` reimplementation.** Delete the numpy helpers.
- [ ] **Step 5 — `make install-dev` (if any C++ needed; likely none); run
  `tests/test_supervised_forecast_pairs.py` — batch regression + graph + lazy + compliance all pass.**
- [ ] **Step 6 — update `docs/functions_supervised/forecast_pairs.md`** to note it runs
  in all regimes; regenerate help.json.
- [ ] **Step 7 — full suite + `make docs`. Commit.**

> If the y-clock selection in duration mode cannot be expressed with existing C++
> nodes (Filter/CombineLatest emit modes), that is a BLOCKER: document exactly which
> node capability is missing and stop for review rather than falling back to numpy.

---

## Task 3: C++ multi-column resample reducer (`ohlc_bars`/`ohlcv_bars` + `ohlcv`/`ohlcv2` retrofit)

**Files:**
- Modify: `include/screamer/dag/resample_params.h` (reducer kinds + per-column plan),
  `include/screamer/dag/resample_node.h` (multi-column accumulator + emit),
  `include/screamer/dag/resample_generic_node.h` (if the generic path needs the plan)
- Modify: `bindings/bindings_streams.cpp` (pass the reducer plan through)
- Modify: `screamer/streams.py` (map `ohlc_bars`/`ohlcv_bars`/`ohlcv`/`ohlcv2` -> plan;
  **delete `_resample_ohlcv` and the two eager-only `raise` sites**; add to `_RESAMPLE_AGGS`)
- Modify: `docs/functions_streams/Resample.md`; test files `tests/test_resample_ohlcv.py`
  (extend for graph/lazy), new `tests/test_resample_bars.py`

**Interfaces:**
- Produces agg strings `'ohlc_bars'`, `'ohlcv_bars'` and makes `'ohlcv'`/`'ohlcv2'`
  all-regime. Reducer plans (internal): ohlc_bars `[first,max,min,last]`; ohlcv_bars
  `[first,max,min,last,sum,sum,...]`; ohlcv `[first,max,min,last,sum]`; ohlcv2
  `[first,max,min,last,sumPos,sumNeg]`.

**Approach:** generalize the reducer. Give the accumulator per-column state (a vector
of the existing scalar accumulators) and carry a `std::vector<ResampleReducer>` plan;
`emit()` writes each column via its reducer. Add `SumPos`/`SumNeg` reducer kinds
(`sum of max(v,0)` / `sum of min(v,0)`). Output width = plan size. The node already
sees full-width frames, so this runs identically in every regime (single pass).

- [ ] **Step 1 — regression test: batch `ohlcv`/`ohlcv2` unchanged.** The existing
  `tests/test_resample_ohlcv.py` assertions must still pass byte-for-byte after the retrofit.
- [ ] **Step 2 — new functional tests** in `tests/test_resample_bars.py`:

```python
import numpy as np
from screamer.streams import Resample

def test_ohlc_bars_reaggregates_built_bars():
    # 6 one-minute OHLC bars -> 2 three-bar bars
    O=np.array([1.,2,3,4,5,6]); H=O+0.5; L=O-0.5; C=O+0.1
    bars = np.column_stack([O,H,L,C])
    out, idx = Resample((bars, np.arange(6)), count=3, agg='ohlc_bars')
    # bar0 = [first O, max H, min L, last C] over rows 0..2
    np.testing.assert_allclose(out[0], [1.0, 3.5, 0.5, 3.1])
    np.testing.assert_allclose(out[1], [4.0, 6.5, 3.5, 6.1])

def test_ohlcv_bars_sums_trailing_columns():
    O=np.array([1.,2,3,4]); H=O+1; L=O-1; C=O; V=np.array([10.,20,30,40])
    bars = np.column_stack([O,H,L,C,V])
    out, _ = Resample((bars, np.arange(4)), count=2, agg='ohlcv_bars')
    np.testing.assert_allclose(out[0], [1, 3, 0, 2, 30])   # V summed
    np.testing.assert_allclose(out[1], [3, 5, 2, 4, 70])

def test_ohlc_bars_rejects_wrong_width():
    import pytest
    with pytest.raises(ValueError):
        Resample((np.ones((4,3)), np.arange(4)), count=2, agg='ohlc_bars')
```

- [ ] **Step 3 — the compliance/all-regime test (the point of this task):**

```python
def test_multicol_aggs_run_in_all_regimes():
    from screamer import Input, Pipeline
    bars = np.column_stack([np.arange(6.), np.arange(6.)+1, np.arange(6.)-1,
                            np.arange(6.), np.ones(6)*10])
    idx = np.arange(6)
    for agg in ('ohlc_bars','ohlcv_bars','ohlcv','ohlcv2'):
        w = 5 if agg in ('ohlcv_bars',) else (4 if agg=='ohlc_bars' else (5 if agg=='ohlcv' else 6))
        b = bars[:, :2] if agg in ('ohlcv','ohlcv2') else bars[:, :4] if agg=='ohlc_bars' else bars
        eager, _ = Resample((b, idx), count=2, agg=agg)
        # graph regime: MUST NOT raise (previously ohlcv/ohlcv2 raised here)
        node = Resample(Input('x'), count=2, agg=agg)
        graph_out = Pipeline([Input('x')], [node])((b, idx))   # exact API per existing tests
        np.testing.assert_allclose(np.asarray(graph_out[0]), eager, equal_nan=True)
```

- [ ] **Step 4 — implement** the multi-column accumulator + `SumPos`/`SumNeg`; map the
  four aggs to plans in `streams.py`; delete `_resample_ohlcv` and the eager-only raises.
- [ ] **Step 5 — `make install-dev`; run `tests/test_resample_ohlcv.py` (regression) +
  `tests/test_resample_bars.py` (functional + all-regime). All pass.**
- [ ] **Step 6 — docs + help.json.** Update `Resample.md`; regenerate.
- [ ] **Step 7 — full suite + `make docs`. Commit.**

---

## Task 4: `Resample` cumulative-driver mode (information bars)

**Files:**
- Modify: `include/screamer/dag/resample_params.h` (`ResampleMode::ByCumulative`,
  `threshold` field), `include/screamer/dag/resample_node.h` (driver input + cumulative
  boundary), `bindings/bindings_streams.cpp`, `screamer/streams.py` (`threshold=` mode +
  driver input), `docs/functions_streams/Resample.md`, new `tests/test_resample_cumulative.py`

**Interfaces:**
- `Resample(value, driver, threshold=T, agg=...)` — `threshold=` is the 4th mutually
  exclusive mode selector; a driver input is required. Bucket closes when cumulative
  driver since the last close `>= threshold`, the crossing observation included, then
  the accumulator resets to 0. `threshold <= 0` -> ValueError; NaN driver ignored (does
  not advance the clock); `threshold=` with any of freq/every/count -> the "exactly one
  of" error; `threshold=` without a driver (or a driver without `threshold=`) -> error.

**Approach:** the node receives `[value, driver]` (width-2 frame in ByCumulative mode;
the Python layer aligns value+driver via the existing combine mechanism). Accumulate
the driver; on `cum >= threshold`, `emit(label)`, reset acc + cum. Value column feeds
the existing (multi-column, from Task 3) reducer.

- [ ] **Step 1 — functional tests** in `tests/test_resample_cumulative.py`:

```python
import numpy as np, pytest
from screamer.streams import Resample

def test_volume_bars_close_on_threshold():
    price  = np.array([10.,11,12,13,14,15])
    volume = np.array([ 4., 3, 5, 2, 6, 1])   # cumsum: 4,7(>=5 close),... 
    out, idx = Resample((price, np.arange(6)), volume, threshold=5, agg='ohlc')
    # first bar closes at the 2nd obs (cum 4->7 crosses 5), includes obs 0..1
    np.testing.assert_allclose(out[0], [10, 11, 10, 11])   # O H L C over obs 0..1

def test_threshold_non_positive_rejected():
    with pytest.raises(ValueError):
        Resample((np.ones(3), np.arange(3)), np.ones(3), threshold=0, agg='sum')

def test_threshold_conflicts_with_count():
    with pytest.raises(ValueError):
        Resample((np.ones(3), np.arange(3)), np.ones(3), threshold=5, count=2, agg='sum')

def test_nan_driver_does_not_advance_clock():
    price  = np.array([10.,11,12,13])
    volume = np.array([ 4.,np.nan,3,10])   # NaN does not add to the clock
    out, _ = Resample((price, np.arange(4)), volume, threshold=6, agg='last')
    # clock: 4, (nan skip)4, 7(>=6 close incl obs0..2), then 10 close
    assert len(out) == 2
```

- [ ] **Step 2 — all-regime test** (eager vs graph vs lazy identical), same shape as Task 3 Step 3.
- [ ] **Step 3 — implement** the `ByCumulative` mode (params + node) and the Python
  `threshold=`/driver dispatch + validation.
- [ ] **Step 4 — `make install-dev`; tests pass.**
- [ ] **Step 5 — docs (`Resample.md` threshold mode) + help.json regenerate.**
- [ ] **Step 6 — full suite + `make docs`. Commit.**

---

## Final (after all 4 tasks)

- [ ] Whole-branch review (dispatch a code-reviewer subagent over the full diff).
- [ ] `python -m pytest -q` green; `make docs` clean.
- [ ] Leave the branch for user review. Do NOT bump the version or merge.

---

## Task 5: `Resample` `clock=` mode (target-clock resample) + `forecast_pairs` duration rewire

Spec: Section 5 of the design doc. Closes the last eager-only path (forecast_pairs duration).

**Files:** `include/screamer/dag/resample_params.h` (`ResampleMode::ByClock`),
`include/screamer/dag/resample_node.h` (`push_by_clock`: clock as 2nd input, bucket
closes at each clock tick over `(prev_tick, tick]`, inclusive ties, label = tick),
`include/screamer/dag/compiled_graph.h` + `bindings/bindings_dag.cpp` (width + clock
wiring), `screamer/streams.py` + `screamer/dag.py` (`clock=` mode selector + validation),
`screamer/supervised.py` (duration rewire), `docs/functions_streams/Resample.md`,
`docs/functions_supervised/forecast_pairs.md`, new `tests/test_resample_clock.py`,
extend `tests/test_supervised_forecast_pairs.py`.

**Interfaces:** `Resample(value, clock, agg='last', fill='carry')` — `clock=` is a 5th
mutually-exclusive mode selector requiring a clock input; the clock contributes only
its event indices. `agg='last'`+`fill='carry'` = as-of. Multi-column value supported
(reuses Task 3 reducer). All-regime, C++-core.

- [ ] **Step 1 — functional tests first** (`tests/test_resample_clock.py`): as-of on a
  hand-worked example where value-clock != target-clock, with a gap that exercises
  `fill='carry'`, a tie at a clock tick, and value events before the first tick;
  `agg='sum'` between clock ticks; multi-column value; the edge rejections (clock= with
  freq/every/count/threshold; clock= without a clock input). Plus an all-regime test
  (eager==graph==lazy) matching the shape in `tests/test_resample_bars.py`.
- [ ] **Step 2 — run, see fail.**
- [ ] **Step 3 — implement `ByClock`** in the node (build on the ByCumulative 2-input /
  wide plumbing): the clock is the 2nd input; at each clock event, close the bucket
  (aggregating value events with index in `(prev_tick, tick]`, inclusive of the tick),
  emit labelled at the tick. Reuse the multi-column reducer + fill machinery. Wire
  `clock=` in `streams.py`/`dag.py` with validation.
- [ ] **Step 4 — `make install-dev`; the clock tests pass.**
- [ ] **Step 5 — rewire `forecast_pairs` duration mode** (`supervised.py`):
  `Resample(Delay(duration)(X), clock=y, agg='last', fill='carry')` paired with y, then
  `Dropna` when requested — a pure Pipeline. DELETE the `frozenset` selection and both
  `NotImplementedError` raises. Add duration-mode all-regime tests (eager==graph==lazy,
  and equal to the current batch output). Confirm `test_no_numpy_datapath_in_supervised`
  passes with duration now included.
- [ ] **Step 6 — docs** (`Resample.md` clock mode + example; `forecast_pairs.md` now
  all-regime). Regenerate help.json.
- [ ] **Step 7 — full suite green + `make docs` clean. Commit** (`feat(resample): clock=
  target-clock mode; forecast_pairs duration all-regime`). NO version bump.
