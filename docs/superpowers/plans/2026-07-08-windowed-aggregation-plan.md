# Windowed Aggregation, Expanding Statistics, and Signed-Part Helpers - Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Generalize `resample` into a full windowed-aggregation operator (`agg = str | functor | dict`), move its per-bucket compute into C++, and add an `Expanding*` statistic family plus `PosPart`/`NegPart` helpers, so arbitrary per-bar statistics (OHLCV, buy/sell volume, trend, skew, ...) are expressible.

**Architecture:** All numeric logic lives in the C++ core under the `EvalOp` interface and is driven by the `CompiledGraph` push-node engine; the eager (array) path stops computing in Python and instead builds a one-node graph and runs it. Python becomes a thin marshalling/dispatch shim so pure-C++ users and a future JS/WASM binding get identical functionality.

**Tech stack:** C++17 + pybind11 (scikit-build-core / CMake build), Python 3.11-3.14, pytest. Design spec: `docs/superpowers/specs/2026-07-08-windowed-aggregation-design.md`.

## Global Constraints

- **C++-first.** No numeric loops in Python. Every reducer/accumulator/statistic is a C++ `EvalOp`; Python only marshals arguments, dispatches by input regime (raw / `Stream` / `Node`), and attaches labels.
- **Causality is a hard rule.** Output depends only on current and past inputs. A bar value emits at bar close (bucket boundary or end-of-input flush), never using future ticks.
- **Batch == streaming == oracle.** Every operator must give identical results eager, in a live-driven `Dag`, and against the test oracle. The existing `tests/test_dag_resample.py` and `tests/test_streams_resample.py` encode this; keep them green.
- **NaN policy `ignore`** for all new accumulators: skip NaN in state, emit NaN at a NaN index where applicable, recover at the next finite sample.
- **Do not hand-edit version strings.** Versioning is `make patch/minor/major` only.
- **Add-a-function checklist** applies to every new functor: C++ class + pybind11 binding + `docs/functions_<family>/<Name>.md` frontmatter + `topics:` from `docs/topics.yml` + `nan_policy` + a `devtools/baselines/<Name>.py` reference where feasible. `tests/test_doc_coverage.py` fails otherwise.
- **Build loop:** after any C++ change run `make build` (compiles the extension into `screamer/` and regenerates `screamer/__init__.py`); run `poetry run pytest -q`.

## Key existing files (from codebase exploration)

- `include/screamer/common/eval_op.h` - `EvalOp`: `n_in()`, `n_out()`, `eval(const double* in, double* out)`, `reset()`.
- `include/screamer/common/base.h` (`ScreamerBase`, 1->1, `process_scalar`), `common/functor_base.h` (`FunctorBase<Derived,N,M>`), `common/transform.h` (`Transform<fn>`), `transform_functions.h` (elementwise fns like `relu`), `common/math.h` (`var_from_stats`, `skew_from_stats`, `kurt_from_stats`, `skew_n_const`, `kurt_n_const`).
- `include/screamer/dag/resample_node.h` (`ResampleNode<Index>`, `ResampleAccum`), `dag/resample_params.h` (`ResampleAgg` enum, `resample_width`), `dag/compiled_graph.h` (`CompiledGraph`, `Sink`, `GatherSink`), `dag/graph.h` (`GraphBuilder`, `NodeKind`), `dag/frame.h` (`Frame`, `Sink`).
- `include/screamer/rolling_var.h` / `rolling_skew.h` / `rolling_kurt.h` / `rolling_poly1.h`, `cum_sum.h` / `cum_max.h` / `cum_min.h` / `cum_prod.h`, `detail/rolling_sum.h`.
- `bindings/bindings_math.cpp`, `bindings/bindings_dag.cpp`, `bindings/bindings_rolling.cpp`, `bindings/bindings.cpp`.
- `screamer/streams.py` (`resample`, `_ResampleAccum` ~lines 537-696), `screamer/dag.py` (`_compile_cpp` resample dispatch ~lines 193-240), `screamer/data/help.json`, `devtools/generate_screamer__init__.py`.
- Tests: `tests/test_streams_resample.py`, `tests/test_dag_resample.py`, `tests/test_doc_coverage.py`, `tests/test_nan_input_compliance.py`, `tests/test_stream_vs_batch.py`.

---

### Task 1: Move the eager resample path onto the C++ engine (behavior-preserving)

Retire the Python `_ResampleAccum` and the eager bucketing loop; route raw-array and `Stream` inputs through the same `CompiledGraph`/`ResampleNode` the graph path already uses. **No public API or result change.**

**Files:**
- Modify: `screamer/streams.py` (`resample`, ~lines 537-696; the eager branch and `_ResampleAccum`)
- Reference (no change): `screamer/dag.py` (`_compile_cpp`, `make_operator_node`), `include/screamer/dag/resample_node.h`, `bindings/bindings_dag.cpp`
- Test: `tests/test_streams_resample.py`, `tests/test_dag_resample.py`

**Interfaces:**
- Produces: `resample(values, index=None, *, every=None, count=None, agg="last", origin=0, label="left")` unchanged signature and return, but the eager path now delegates to C++.

- [ ] **Step 1: Characterization test lock.** Run the existing resample suites and record current outputs as the oracle.
  Run: `poetry run pytest tests/test_streams_resample.py tests/test_dag_resample.py -q`
  Expected: PASS (this is the behavior to preserve).
- [ ] **Step 2: Add an eager-through-C++ helper.** In `screamer/streams.py`, add a private `_resample_eager_via_cpp(values_2d_or_1d, index, params)` that builds a minimal graph (one `Input` -> one `Resample` node -> gather) and calls `run_batch`, reusing the same compile path `Dag` uses. Do not write the numeric loop.
- [ ] **Step 3: Route the eager branch.** Replace the body of the eager branch in `resample` (the `for k, v in zip(...)` loops and `_ResampleAccum` usage) with a call to `_resample_eager_via_cpp`. Keep the `_adapt`/return-regime logic.
- [ ] **Step 4: Delete dead code.** Remove `_ResampleAccum` and the Python bucketing loops. Grep to confirm no other references: `git grep -n _ResampleAccum`.
- [ ] **Step 5: Run the suites.**
  Run: `make build && poetry run pytest tests/test_streams_resample.py tests/test_dag_resample.py tests/test_stream_vs_batch.py -q`
  Expected: PASS, identical results (batch == stream == oracle preserved).
- [ ] **Step 6: Full suite + commit.**
  Run: `poetry run pytest -q` (expect the full suite green)
  Commit: `refactor(resample): eager path runs on the C++ engine; retire Python accumulator`

---

### Task 2: `PosPart` / `NegPart` elementwise functors

Small, independent. `PosPart(x)=max(x,0)` (same as `relu`), `NegPart(x)=max(-x,0)`.

**Files:**
- Modify: `include/screamer/transform_functions.h` (add `pos_part`, `neg_part`)
- Modify: `bindings/bindings_math.cpp` (add `PosPart`, `NegPart` bindings)
- Create: `docs/functions_math/PosPart.md`, `docs/functions_math/NegPart.md`
- Create: `devtools/baselines/PosPart.py`, `devtools/baselines/NegPart.py`
- Test: `tests/test_pos_neg_part.py`

**Interfaces:**
- Produces: `screamer.PosPart`, `screamer.NegPart` (1->1 functors, `nan_policy: ignore`).

- [ ] **Step 1: Failing test.**
```python
# tests/test_pos_neg_part.py
import numpy as np
from screamer import PosPart, NegPart

def test_pos_neg_part_matches_numpy():
    x = np.array([-2.0, -0.5, 0.0, 1.5, np.nan, 3.0])
    np.testing.assert_array_equal(np.asarray(PosPart()(x)), np.where(np.isnan(x), x, np.maximum(x, 0.0)))
    np.testing.assert_array_equal(np.asarray(NegPart()(x)), np.where(np.isnan(x), x, np.maximum(-x, 0.0)))
    # decomposition identity: x == PosPart(x) - NegPart(x) on finite values
    fin = x[np.isfinite(x)]
    np.testing.assert_allclose(np.asarray(PosPart()(fin)) - np.asarray(NegPart()(fin)), fin)
```
- [ ] **Step 2: Run it, expect ImportError/fail.** `poetry run pytest tests/test_pos_neg_part.py -q`
- [ ] **Step 3: Implement the functions** in `include/screamer/transform_functions.h`:
```cpp
inline double pos_part(double x) { return std::isnan(x) ? x : (x > 0.0 ? x : 0.0); }
inline double neg_part(double x) { return std::isnan(x) ? x : (x < 0.0 ? -x : 0.0); }
```
- [ ] **Step 4: Bind them** in `bindings/bindings_math.cpp` (mirror the `Relu` block exactly, class names `PosPart` / `NegPart`, binding `__call__` and `reset`).
- [ ] **Step 5: Docs + baselines.** Write the two `docs/functions_math/*.md` pages (frontmatter: `implementation_family: math`, `topics: [arithmetic]`, `inputs: 1`, `outputs: 1`, `parameters: []`, `nan_policy: ignore`) and the two baseline classes (`PosPart_numpy`, `NegPart_numpy`).
- [ ] **Step 6: Build, test, regenerate help.**
  Run: `make build && poetry run python devtools/build_help_registry.py && poetry run pytest tests/test_pos_neg_part.py tests/test_doc_coverage.py -q`
  Expected: PASS.
- [ ] **Step 7: Commit.** `feat(math): PosPart / NegPart elementwise functors`

---

### Task 3: `Expanding*` statistic family + `Cum*` aliases

`ExpandingMean/Var/Std/Skew/Kurt/Slope` (whole-history, resettable). Each keeps plain unbounded running sums and reuses `common/math.h` helpers - no window buffer, no `start_policy`. `Expanding{Sum,Max,Min,Prod}` are thin aliases to the existing `Cum*`.

**Files (per stat, mirror the rolling analogue):**
- Create: `include/screamer/expanding_mean.h`, `expanding_var.h`, `expanding_std.h`, `expanding_skew.h`, `expanding_kurt.h`, `expanding_slope.h`
- Create: `include/screamer/expanding_sum.h`, `expanding_max.h`, `expanding_min.h`, `expanding_prod.h` (aliases wrapping `Cum*`)
- Create: `bindings/bindings_expanding.cpp`; register it in `bindings/bindings.cpp` (`init_bindings_expanding(m)`) and in `CMakeLists.txt` if binding files are listed there
- Create: `docs/functions_<family>/Expanding*.md` (family: choose `misc` or a new grouping; topics per stat, e.g. `statistics`), and `devtools/baselines/Expanding*.py` (pandas `.expanding()` references)
- Test: `tests/test_expanding.py`

**Interfaces:**
- Produces: `screamer.ExpandingMean/Var/Std/Skew/Kurt/Slope/Sum/Max/Min/Prod`, all 1->1 `ScreamerBase` functors with `reset()`, `nan_policy: ignore`.

- [ ] **Step 1: Failing tests vs pandas `.expanding()`.**
```python
# tests/test_expanding.py
import numpy as np, pandas as pd
from screamer import ExpandingMean, ExpandingVar, ExpandingStd, ExpandingSkew, ExpandingKurt

def _pd(x, method):
    return getattr(pd.Series(x).expanding(), method)().to_numpy()

def test_expanding_moments_match_pandas():
    rng = np.random.default_rng(0); x = rng.normal(size=200)
    np.testing.assert_allclose(np.asarray(ExpandingMean()(x)), _pd(x, "mean"), rtol=1e-10, equal_nan=True)
    np.testing.assert_allclose(np.asarray(ExpandingVar()(x)),  _pd(x, "var"),  rtol=1e-9,  equal_nan=True)
    np.testing.assert_allclose(np.asarray(ExpandingStd()(x)),  _pd(x, "std"),  rtol=1e-9,  equal_nan=True)
    np.testing.assert_allclose(np.asarray(ExpandingSkew()(x)), _pd(x, "skew"), rtol=1e-8,  equal_nan=True)
    np.testing.assert_allclose(np.asarray(ExpandingKurt()(x)), _pd(x, "kurt"), rtol=1e-8,  equal_nan=True)

def test_reset_restarts_accumulation():
    x = np.array([1.0, 2.0, 3.0]); m = ExpandingMean()
    _ = m(x); m.reset()
    assert m(np.array([10.0]))[0] == 10.0
```
  Note: confirm pandas' ddof/bias conventions against screamer's `Rolling*`; pin any documented divergence with a comment (as the repo already does for other stats).
- [ ] **Step 2: Run, expect fail.** `poetry run pytest tests/test_expanding.py -q`
- [ ] **Step 3: Implement `ExpandingVar` (template for the moment stats).** New `include/screamer/expanding_var.h`, class `ExpandingVar : public ScreamerBase`, members `double sum_x_=0, sum_xx_=0; long n_=0;`. `process_scalar(x)`: if NaN, return NaN (do not update); else `sum_x_+=x; sum_xx_+=x*x; ++n_;` and `return var_from_stats(sum_x_, sum_xx_, n_)` (use the same helper `RollingVar` uses; match its ddof). `reset()` zeroes the members. Mirror `include/screamer/rolling_var.h`'s formula, dropping the windowed `detail::RollingSum` in favor of the running sums.
- [ ] **Step 4: Implement the rest by the same pattern.** `ExpandingMean` (sum_x, n), `ExpandingStd` (`sqrt` of the var), `ExpandingSkew` (sum_x, sum_xx, sum_xxx + `skew_from_stats`/`skew_n_const`), `ExpandingKurt` (four sums + `kurt_from_stats`/`kurt_n_const`), `ExpandingSlope` (closed-form OLS from `rolling_poly1.h` with `x = 0..n-1` implicit, keeping `sum_y_`, `sum_xy_`, `n_`, no `DelayBuffer`).
- [ ] **Step 5: Implement the `Cum*` aliases.** `ExpandingSum/Max/Min/Prod` as thin subclasses (or `using`) of `CumSum/CumMax/CumMin/CumProd`; bind under the `Expanding*` names.
- [ ] **Step 6: Bind + register.** `bindings/bindings_expanding.cpp` with one `py::class_<...>` per class (mirror `bindings_rolling.cpp`'s `.def(py::init<>()).def("__call__", ...).def("reset", ...)`), add `init_bindings_expanding(m)` to `bindings/bindings.cpp`, and add the new `.cpp` to the build source list.
- [ ] **Step 7: Docs + baselines** for each new functor (frontmatter + topics + `nan_policy: ignore` + pandas baseline).
- [ ] **Step 8: Build + test.**
  Run: `make build && poetry run python devtools/build_help_registry.py && poetry run pytest tests/test_expanding.py tests/test_doc_coverage.py tests/test_nan_input_compliance.py -q`
- [ ] **Step 9: Commit.** `feat(expanding): ExpandingMean/Var/Std/Skew/Kurt/Slope + Cum* aliases`

---

### Task 4: `agg` accepts an arbitrary functor (generic reducer node)

Add a C++ `GenericResampleNode<Index>` that owns an `EvalOp*` reducer: feed each in-bar sample via `eval`, remember the latest output, emit it at bucket close, then `reset()` the reducer. Wire `resample(..., agg=<functor>)` through it.

**Files:**
- Create: `include/screamer/dag/resample_generic_node.h` (`GenericResampleNode<Index>`)
- Modify: `include/screamer/dag/resample_params.h` (carry an optional reducer handle), `dag/compiled_graph.h` (instantiate generic vs builtin at the Resample case), `bindings/bindings_dag.cpp` (`add_resample` accepts an optional `EvalOp*`/`py::object`)
- Modify: `screamer/dag.py` (`_compile_cpp` resample dispatch: string -> enum, functor -> pass the op), `screamer/streams.py` (`resample` accepts `agg` as a functor; `_RESAMPLE_AGGS` validation updated)
- Test: `tests/test_resample_functor_agg.py`

**Interfaces:**
- Consumes: `EvalOp` (Task 3 functors), the eager-through-C++ path (Task 1).
- Produces: `resample(values, ..., agg=<EvalOp functor>)` returning one value (or an `n_out()`-wide row) per bar; `batch == stream`.

- [ ] **Step 1: Failing equivalence test.** A functor reducer that duplicates a builtin must match it, and an expanding reducer must give the whole-bar statistic.
```python
# tests/test_resample_functor_agg.py
import numpy as np
from screamer import ExpandingSum, ExpandingSkew
from screamer.streams import resample

def test_functor_sum_equals_builtin_sum():
    x = np.arange(20.0); idx = np.arange(20, dtype=np.int64)
    a, ia = resample(x, idx, every=5, agg="sum")
    b, ib = resample(x, idx, every=5, agg=ExpandingSum())
    np.testing.assert_array_equal(np.asarray(ia), np.asarray(ib))
    np.testing.assert_allclose(np.asarray(a), np.asarray(b))

def test_functor_reducer_resets_each_bar():
    # skew over each bar independently; a functor reset per bar must not leak across bars
    x = np.concatenate([np.zeros(5), np.arange(5.0)]); idx = np.arange(10, dtype=np.int64)
    vals, _ = resample(x, idx, every=5, agg=ExpandingSkew())
    assert vals.shape[0] == 2  # two bars, independent
```
- [ ] **Step 2: Run, expect fail** (agg=functor unsupported). `poetry run pytest tests/test_resample_functor_agg.py -q`
- [ ] **Step 3: Implement `GenericResampleNode<Index>`.** Mirror `resample_node.h`'s bucketing (`push_by_index`/`push_by_count`/`flush`), but instead of `ResampleAccum`, hold `EvalOp* reducer_` and a `std::vector<double> latest_(reducer_->n_out())` plus a `bool has_`. On each in-bar sample: `reducer_->eval(&v, latest_.data()); has_=true;`. At bucket close: emit `latest_` (width `reducer_->n_out()`), then `reducer_->reset(); has_=false;`. Preserve NaN-ignore (skip feeding NaN, matching `ResampleAccum`).
- [ ] **Step 4: Thread the reducer through params + compile.** Add an optional reducer pointer to `ResampleParams` (or a sibling field); in `compiled_graph.h` at the Resample case, instantiate `GenericResampleNode` when a reducer is present, else the existing `ResampleNode`. Extend `bindings_dag.cpp` `add_resample` to accept an optional op handle (an `EvalOp*` extracted from the pybind object).
- [ ] **Step 5: Python dispatch.** In `screamer/dag.py` `_compile_cpp`, if `agg` is an `EvalOp` (not a string), pass the op to `add_resample` instead of an enum code. In `screamer/streams.py`, accept `agg` as a functor (skip the string validation for functor aggs) and, for the eager path, route through the same C++ builder (Task 1's helper), passing the functor.
- [ ] **Step 6: Build + test.**
  Run: `make build && poetry run pytest tests/test_resample_functor_agg.py tests/test_streams_resample.py tests/test_dag_resample.py -q`
  Expected: PASS (builtin string aggs unaffected; functor aggs work; batch == stream).
- [ ] **Step 7: Commit.** `feat(resample): agg accepts any EvalOp functor (generic reducer node)`

---

### Task 5: `Stream.columns` + unify all `resample` returns on `Stream`

Give `Stream` an optional `columns` tuple and named access; make `resample` always return a `Stream` (values + index + optional columns) across raw / Stream / Node regimes.

**Files:**
- Modify: `screamer/streams.py` (`Stream` class: add `columns`, `__getitem__(name)`, `column(name)`; `resample` return path)
- Modify: any callers/tests that unpack `resample(...)` as a 2-tuple
- Test: `tests/test_stream_columns.py`, update `tests/test_streams_resample.py`

**Interfaces:**
- Produces: `Stream` with `.values` (1-D or 2-D), `.index`, `.columns` (tuple or None), `stream["name"]` -> 1-D view. `resample(...) -> Stream` in all regimes.

- [ ] **Step 1: Failing test.**
```python
# tests/test_stream_columns.py
import numpy as np
from screamer.streams import resample, Stream

def test_resample_returns_stream_with_columns():
    x = np.arange(20.0); idx = np.arange(20, dtype=np.int64)
    bars = resample(x, idx, every=5, agg="ohlc")
    assert isinstance(bars, Stream)
    assert tuple(bars.columns) == ("open", "high", "low", "close")
    np.testing.assert_allclose(bars["open"], bars.values[:, 0])
```
- [ ] **Step 2: Run, expect fail.** `poetry run pytest tests/test_stream_columns.py -q`
- [ ] **Step 3: Extend `Stream`.** Add `columns=None` to construction; `__getitem__`/`column` return the labelled 1-D column (raise a clear error if unlabelled or name missing).
- [ ] **Step 4: Unify `resample` returns.** Always return a `Stream`; attach `columns` for multi-column aggs (`ohlc` -> `("open","high","low","close")`; single-value aggs -> `None`). The C++ node returns the matrix + width; Python attaches names (labels are marshalling, not compute).
- [ ] **Step 5: Fix callers.** Update the resample suite and any code unpacking `(values, index)` to use `Stream`. `git grep -n "resample(" -- tests` to find them.
- [ ] **Step 6: Build + full suite.**
  Run: `make build && poetry run pytest -q`
- [ ] **Step 7: Commit.** `feat(streams): Stream.columns + unify resample returns on Stream`

---

### Task 6: `agg` as a dict of reducers (labelled multi-column bars)

`resample(x, every=BAR, agg={"buy_vol": ..., "skew": ...})` runs each reducer over the same bucketing and returns one labelled `Stream` (columns = dict keys). Python orchestrates composition + labels; each reducer runs in C++.

**Files:**
- Modify: `screamer/streams.py` (`resample`: dict branch), reuse the Task 4 functor path and Task 5 `Stream.columns`
- Test: `tests/test_resample_dict_agg.py`

**Interfaces:**
- Consumes: Task 4 (functor reducers), Task 5 (`Stream.columns`).
- Produces: `resample(..., agg={name: str|functor})` -> `Stream` with `.columns == tuple(dict)`, column order = insertion order.

- [ ] **Step 1: Failing test.**
```python
# tests/test_resample_dict_agg.py
import numpy as np
from screamer import ExpandingSkew
from screamer.streams import resample

def test_dict_agg_labelled_columns():
    x = np.arange(20.0); idx = np.arange(20, dtype=np.int64)
    bars = resample(x, idx, every=5, agg={"total": "sum", "skew": ExpandingSkew()})
    assert tuple(bars.columns) == ("total", "skew")
    total, _ = resample(x, idx, every=5, agg="sum")
    np.testing.assert_allclose(bars["total"], total.values if hasattr(total, "values") else total)
```
- [ ] **Step 2: Run, expect fail.** `poetry run pytest tests/test_resample_dict_agg.py -q`
- [ ] **Step 3: Implement the dict branch.** For each `(name, agg)`, build a resample over the shared bucketing (same `every`/`count`/`origin`), all sharing the bar index; horizontally stack the per-reducer columns into one 2-D matrix and set `columns` from the keys. Multi-column reducers (an `ohlc` inside a dict) expand to prefixed columns (`name.open`, ...) or reject for v1 (document the choice). Keep composition in Python; compute stays in the C++ reducers.
- [ ] **Step 4: Build + test.**
  Run: `make build && poetry run pytest tests/test_resample_dict_agg.py tests/test_streams_resample.py -q`
- [ ] **Step 5: Commit.** `feat(resample): dict agg -> labelled multi-column bars`

---

### Task 7: `ohlcv` / `ohlcv2` multi-input string aggs

`ohlcv` -> `[open,high,low,close,volume]` (input `[price, volume]`); `ohlcv2` -> `[open,high,low,close,buy_vol,sell_vol]` (input `[price, signed_volume]`). Implement as multi-input C++ reducer functors reused through the Task 4 generic path.

**Files:**
- Create: `include/screamer/dag/ohlcv_reducer.h` (two `FunctorBase`-based reducers: `OhlcvReducer` 2->5, `Ohlcv2Reducer` 2->6, each expanding + resettable)
- Modify: `bindings/bindings_dag.cpp` (expose them or register as builtin reducers), `screamer/streams.py` (`_RESAMPLE_AGGS` + mapping so the `ohlcv`/`ohlcv2` strings resolve to these reducers), `screamer/dag.py` (compile mapping)
- Modify: multi-input handling in the resample node so a 2-column input reaches a 2-input reducer (uses the existing "(T,N) as N inputs" model)
- Test: `tests/test_resample_ohlcv.py`

**Interfaces:**
- Consumes: Task 4 generic reducer path; the `(T,N)`-as-N-inputs functor convention.
- Produces: `resample([price, volume], every=BAR, agg="ohlcv")` and `resample([price, signed_volume], ..., agg="ohlcv2")`, labelled `Stream`s.

- [ ] **Step 1: Failing test** comparing against `resample(price, agg="ohlc")` for the OHLC columns and `PosPart`/`NegPart` + sum for the volume columns:
```python
# tests/test_resample_ohlcv.py
import numpy as np
from screamer import PosPart, NegPart
from screamer.streams import resample

def test_ohlcv2_matches_composition():
    price = np.arange(20.0); vol = np.where(np.arange(20) % 2, 1.0, -1.0); idx = np.arange(20, dtype=np.int64)
    bars = resample(np.column_stack([price, vol]), idx, every=5, agg="ohlcv2")
    assert tuple(bars.columns) == ("open","high","low","close","buy_vol","sell_vol")
    o = resample(price, idx, every=5, agg="ohlc")
    np.testing.assert_allclose(bars.values[:, :4], o.values)
    buy, _  = resample(np.asarray(PosPart()(vol)), idx, every=5, agg="sum")
    np.testing.assert_allclose(bars["buy_vol"], buy.values if hasattr(buy, "values") else buy)
```
- [ ] **Step 2: Run, expect fail.** `poetry run pytest tests/test_resample_ohlcv.py -q`
- [ ] **Step 3: Implement the reducers** (`FunctorBase`-derived, expanding within the bar): `OhlcvReducer` tracks first/max/min/last of column 0 and sum of column 1; `Ohlcv2Reducer` tracks OHLC of column 0 and pos/neg sums of column 1. `reset()` clears all.
- [ ] **Step 4: Multi-input wiring.** Ensure the generic resample node feeds a 2-column input row (`const double* in`, width 2) to the 2-input reducer's `eval`. Extend the eager/graph input handling so a `(T,2)` array or a 2-column Stream is accepted for these aggs.
- [ ] **Step 5: String mapping + labels.** Map `"ohlcv"`/`"ohlcv2"` to these reducers, with column labels `("open","high","low","close","volume")` / `(...,"buy_vol","sell_vol")`.
- [ ] **Step 6: Build + test.**
  Run: `make build && poetry run pytest tests/test_resample_ohlcv.py tests/test_streams_resample.py tests/test_dag_resample.py -q`
- [ ] **Step 7: Commit.** `feat(resample): ohlcv / ohlcv2 multi-input bar aggregations`

---

### Task 8: Docs, resample reference, and a bars recipe

Consolidate documentation and add a runnable recipe that exercises the whole feature.

**Files:**
- Modify: `docs/functions_streams/resample.md` (document `agg = str | functor | dict`, the new strings, labelled output, `Stream.columns`)
- Create: a recipe notebook, e.g. `docs/notebooks/*-bars-and-custom-aggregations.ipynb` (tick -> OHLCV, buy/sell vol via `PosPart`/`NegPart`, a custom per-bar `ExpandingSkew`/`ExpandingSlope`)
- Modify: `docs/topics.yml` only if a new topic is warranted (likely not; reuse `streams`, `statistics`, `arithmetic`)
- Modify: `CHANGELOG.md` (user-facing: new functors, resample generalization)
- Test: `poetry run pytest --nbmake docs/notebooks/<new>.ipynb` (notebook executes)

- [ ] **Step 1: Update the resample page** with the generalized `agg` and labelled output, including a short OHLCV and a custom-reducer example.
- [ ] **Step 2: Write the recipe notebook** (seeded, deterministic; it executes at docs build time).
- [ ] **Step 3: Regenerate + build docs.**
  Run: `poetry run python devtools/build_help_registry.py && poetry run python devtools/build_topic_pages.py && make docs`
  Expected: docs build clean; `tests/test_doc_coverage.py` green.
- [ ] **Step 4: CHANGELOG + commit.** `docs: windowed aggregation, expanding stats, PosPart/NegPart`

---

## Deferred (separate plan)

- Comparison family (`Gt/Lt/Ge/Le/Eq`, `0/1` output; `CrossOver`/`CrossUnder`).
- `SumWhere(value, mask)` (2-input reducer, `mask != 0`), which depends on the comparison family.

## Self-review

- **Spec coverage:** `resample(agg=str|functor|dict)` (Tasks 4, 6), `ohlcv`/`ohlcv2` (Task 7), `Expanding*` + `Cum*` aliases (Task 3), `PosPart`/`NegPart` (Task 2), labelled output + `Stream.columns` + unify-on-Stream (Task 5), C++-first (Task 1 moves compute to C++; every functor is an `EvalOp`). All spec items map to a task.
- **Ordering/risk:** Task 1 is behavior-preserving and de-risks the engine move first. Tasks 2-3 are independent functor additions. Task 4 is the core generalization and depends on 1 (+ real reducers from 3 to test). 5 precedes 6/7 (labelled returns). 7 needs multi-input, the most involved builtin, last. 8 consolidates.
- **Types are real:** every referenced type (`EvalOp`, `ScreamerBase`, `FunctorBase`, `ResampleNode`, `CompiledGraph`, `Transform`, `var_from_stats`, `Cum*`) was confirmed in the codebase; new types (`GenericResampleNode`, `Expanding*`, `Ohlcv*Reducer`) are introduced with their file and interface.
- **Interface consistency:** `resample` signature is unchanged except `agg` widening (str -> str|functor|dict) and return unifying on `Stream`; both are pre-1.0 and covered by Tasks 4-6.
- **Open item to confirm during Task 6:** how a multi-column reducer nested inside a `dict` agg expands its labels (prefix vs reject); decide and document in that task.
