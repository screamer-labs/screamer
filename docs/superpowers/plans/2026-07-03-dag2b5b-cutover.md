# DAG-2b-5b — dag.py cutover to the C++ engine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the public `Dag` execute through the C++ engine (compute in C++, batch and live streaming), relocate DAG-1's Python executor to a test-only reference oracle, and prove batch == stream == oracle across a graph matrix. This completes DAG-2: define once, run live, one C++ implementation.

**Architecture:** `Dag.__init__` walks the `Node` graph and builds a C++ `GraphSpec` via `_GraphBuilder` (functors → `add_functor`; `combine_latest` → `add_combine_latest`), compiling to a persistent `_CompiledGraph`. `Dag.__call__` runs the C++ engine (`run_batch`) → M independent output streams → the existing Python `align_outputs` boundary step (a once-per-run numpy op, not per-event compute). `Dag.stream()` drives the engine live via `push_event`. Graph node ops are C++-only, so `combine_latest` in a graph is alignment-only (`func=` dropped). The old full-Python executor becomes `tests/_dag_oracle.py`.

**Tech Stack:** Python, C++ engine (already built), numpy, pytest.

## Global Constraints

- **Compute in C++; align at the boundary in Python.** The C++ engine returns M independent output streams; `align_outputs=True` applies the *existing* `combine_latest`+dedup step (once per run, numpy-vectorized) in Python. This is boundary marshalling, not per-event compute — acceptable outside the C++ core.
- **Graph node ops are C++-only:** `combine_latest` in a graph is alignment-only; passing `func=` with `Node` args raises a clear error (use `Sub`/`Add`/… ). Only functors and `combine_latest` are graph combinators; `merge`/`dropna`/`filter`/`split` as graph nodes raise "not supported as a graph node".
- **Byte-identity:** `Dag` batch output == `Dag` stream output == the DAG-1 reference oracle, across chains/fan-out/multi-output/combine graphs (`np.testing.assert_array_equal`, NaN-aware).
- **The Python `_run` compute executor is removed from production** `dag.py` and relocated to `tests/_dag_oracle.py`; the shared `align_outputs` helper is used by both the new `Dag` and the oracle.
- Never hand-edit `screamer/__init__.py` or version files.
- Build: `make build` **then `make install-dev`** (no C++ change here, but rebuild if stale). Tests: `poetry run pytest tests/test_dag_exec.py tests/test_dag_build.py tests/test_dag_stream.py tests/test_dag_identity.py -v`.

---

## File Structure

- `screamer/dag.py` (modify) — `Dag` compiles to the C++ engine + runs it; extract `_align_results`; remove the Python compute `_run`.
- `screamer/streams.py` (modify) — `combine_latest` graph-mode rejects `func=`; `merge`/`dropna`/`filter`/`split` graph nodes remain recorded but the compiler rejects them (or reject at build).
- `tests/_dag_oracle.py` (create) — the DAG-1 full-Python executor (the old `_run`), used as the reference oracle.
- `tests/test_dag_exec.py`, `tests/test_dag_build.py` (modify) — migrate `func=` graph usages to `Sub`/`Add`.
- `tests/test_dag_identity.py` (create) — batch == stream == oracle matrix.

---

### Task 1: cutover `Dag` to the C++ engine (batch)

**Files:**
- Modify: `screamer/dag.py`, `screamer/streams.py`
- Create: `tests/_dag_oracle.py`
- Modify: `tests/test_dag_exec.py`, `tests/test_dag_build.py`

**Interfaces:**
- Consumes: `screamer_bindings._GraphBuilder` (`add_input`/`add_functor`/`add_combine_latest`/`set_outputs`/`compile`) and `_CompiledGraph.run_batch` (DAG-2b-4/5a).
- Produces: `Dag.__call__(feeds) -> single or tuple of (keys, values)` computed by the C++ engine, aligned in Python; `screamer.dag._align_results(results, align_outputs)`; `tests/_dag_oracle.py::run_oracle(dag_spec, feeds)`.

- [ ] **Step 1: Extract the align helper + write the migrated/oracle tests**

In `screamer/dag.py`, extract the align logic (current `_run` lines that do `combine_latest(*results)` + dedup) into a module-level helper:

```python
def _align_results(results, align_outputs):
    """Boundary align: single stream for M==1; tuple of independent streams for
    align_outputs=False; co-indexed tuple (combine_latest + one-row-per-key) for
    align_outputs=True. Operates on already-computed (keys, values) output streams.
    """
    if len(results) == 1:
        return results[0]
    if not align_outputs:
        return tuple(results)
    from .streams import combine_latest
    aligned_keys, aligned = combine_latest(*results, emit="when_all")
    _, inv_idx = np.unique(aligned_keys[::-1], return_index=True)
    last_idx = np.sort(len(aligned_keys) - 1 - inv_idx)
    aligned_keys = aligned_keys[last_idx]
    aligned = aligned[last_idx]
    return tuple((aligned_keys, aligned[:, j]) for j in range(len(results)))
```

Create `tests/_dag_oracle.py` containing the DAG-1 full-Python executor (copy the current `_run` body, using `screamer.dag._align_results` for the align step):

```python
"""DAG-1 reference oracle — the original pure-Python executor, kept for tests."""
import numpy as np
from screamer.dag import is_node, _align_results
from screamer.dag import _as_stream  # (export it from dag.py if not already module-level)


def run_oracle(dag, feeds):
    memo = {}

    def ev(node):
        k = id(node)
        if k in memo:
            return memo[k]
        op = node.op
        if isinstance(op, tuple) and op[0] == "input":
            result = _as_stream(feeds[op[1]])
        elif isinstance(op, tuple) and op[0] == "combinator":
            fn, kwargs = op[1], op[2]
            result = fn(*[ev(i) for i in node.inputs], **kwargs)
        else:
            ins = [ev(i) for i in node.inputs]
            result = (ins[0][0], op(*[v for (_, v) in ins]))
        memo[k] = result
        return result

    results = [ev(o) for o in dag.outputs]
    return _align_results(results, dag.align_outputs)
```

Migrate the `func=` graph usages: in `tests/test_dag_exec.py::test_align_outputs_default_coindexes_different_branches`, replace `combine_latest(a, b, func=lambda p, q: p - q)` with `Sub()(combine_latest(a, b))` and `combine_latest(a, c, func=lambda p, q: p + q)` with `Add()(combine_latest(a, c))` (import `Sub`, `Add`). In `tests/test_dag_build.py`, `combine_latest(a, b, func=None)` stays valid (func=None is allowed — alignment-only).

- [ ] **Step 2: Run tests to verify current state**

Run: `poetry run pytest tests/test_dag_exec.py -v`
Expected: the migrated `test_align_outputs...` now uses `Sub`/`Add`; it still runs on the OLD Python `_run` at this step (cutover in Step 3), so it should PASS on the old executor after migration. Other tests unchanged.

- [ ] **Step 3: Cut `Dag` over to the C++ engine**

In `screamer/streams.py`, make `combine_latest` reject `func=` in graph mode:

```python
def combine_latest(*series, emit="when_all", func=None):
    if any(is_node(s) for s in series):
        if func is not None:
            raise ValueError(
                "combine_latest(func=...) is not supported in a DAG graph "
                "(graph ops are C++-only); apply a C++ functor to the aligned "
                "output instead, e.g. Sub()(combine_latest(a, b))")
        return make_combinator_node(combine_latest, series, {"emit": emit, "func": None})
    ...  # existing eager body unchanged
```

In `screamer/dag.py`, replace `Dag._run` with a C++-engine build + execute. In `__init__`, after validation, compile to the C++ engine:

```python
    def __init__(self, inputs, outputs, align_outputs=True):
        ...  # existing validation unchanged, through self._names
        self._cg, self._input_order = self._compile_cpp()

    def _compile_cpp(self):
        from . import screamer_bindings as _b
        gb = _b._GraphBuilder()
        ids = {}   # id(node) -> C++ node id

        def build(node):
            key = id(node)
            if key in ids:
                return ids[key]
            op = node.op
            if isinstance(op, tuple) and op[0] == "input":
                nid = gb.add_input()
            elif isinstance(op, tuple) and op[0] == "combinator":
                fn, kwargs = op[1], op[2]
                if getattr(fn, "__name__", "") != "combine_latest":
                    raise ValueError(
                        f"{fn.__name__} is not supported as a DAG graph node")
                inp = [build(i) for i in node.inputs]
                nid = gb.add_combine_latest(inp, kwargs.get("emit") == "when_all")
            else:                                    # functor instance (EvalOp)
                inp = [build(i) for i in node.inputs]
                nid = gb.add_functor(op, inp)
            ids[key] = nid
            return nid

        # build Input nodes first so their ids follow signature order
        for n in self.inputs:
            build(n)
        out_ids = [build(o) for o in self.outputs]
        gb.set_outputs(out_ids)
        # map signature order -> the add_input order (they match: inputs built first)
        return gb.compile(), list(self._names)

    def __call__(self, *args, **kwargs):
        feeds = self._bind_args(args, kwargs)      # {name: feed}
        streams = [_as_stream(feeds[nm]) for nm in self._input_order]
        results = self._cg.run_batch(streams)      # M independent (keys, values2d)
        results = [(k, v.reshape(-1) if v.shape[1] == 1 else v) for (k, v) in results]
        return _align_results(results, self.align_outputs)
```

Delete the old `Dag._run` method. (`_as_stream` must be module-level and importable — it already is.)

Note: `run_batch` returns each output as `(keys, values2d)` of shape `(M,1)` for a 1-wide functor output; reshape width-1 outputs to 1-D so `_align_results`/single-return match the DAG-1 shape contract.

- [ ] **Step 4: Build/run + migrate**

Run: `make build && make install-dev && poetry run pytest tests/test_dag_exec.py tests/test_dag_build.py -v`
Expected: all pass — `Dag` now computes via the C++ engine; the migrated align test passes; `func=`-on-node now raises (add/keep a test asserting that).

- [ ] **Step 5: Full-suite guard + commit**

Run: `poetry run pytest -q` (green), then:

```bash
git add screamer/dag.py screamer/streams.py tests/_dag_oracle.py tests/test_dag_exec.py tests/test_dag_build.py
git commit -m "feat(dag): Dag executes via the C++ engine; _run -> test oracle"
```

---

### Task 2: `Dag.stream()` + batch == stream == oracle matrix

**Files:**
- Modify: `screamer/dag.py`
- Test: `tests/test_dag_identity.py`

**Interfaces:**
- Consumes: `_CompiledGraph.push_event`/`drain`/`reset` (2b-5a); `_dag_oracle.run_oracle`.
- Produces: `Dag.stream(feeds) -> tuple of (keys, values)` — drives the engine live (merge feeds by key, push_event each), then aligns; identical to batch.

- [ ] **Step 1: Write the failing identity matrix**

Create `tests/test_dag_identity.py`:

```python
import numpy as np
import pytest
from screamer import RollingMean, Diff, Sub, Add, Input, Dag, combine_latest
from tests._dag_oracle import run_oracle


def _row(v):
    v = np.ascontiguousarray(v, dtype=np.float64)
    return np.arange(v.size, dtype=np.int64), v


def _series(size, seed):
    rng = np.random.default_rng(seed)
    k = np.sort(rng.integers(0, size * 3, size=size)).astype(np.int64)
    return k, rng.standard_normal(size)


def _chain():
    x = Input("x")
    return Dag(inputs=[x], outputs=[Diff(1)(RollingMean(5)(x))]), [_row(np.random.default_rng(0).standard_normal(150))]


def _fanout():
    x = Input("x")
    s = RollingMean(5)(x)
    return Dag(inputs=[x], outputs=[Diff(1)(s), RollingMean(3)(s)]), [_row(np.random.default_rng(1).standard_normal(150))]


def _combine():
    a, b = Input("a"), Input("b")
    z = RollingMean(4)(Sub()(combine_latest(a, b)))
    return Dag(inputs=[a, b], outputs=[z]), [_series(120, 2), _series(120, 3)]


@pytest.mark.parametrize("factory", [_chain, _fanout, _combine])
def test_batch_equals_oracle(factory):
    dag, feeds = factory()
    got = dag(*feeds)
    exp = run_oracle(dag, {nm: f for nm, f in zip(dag._names, feeds)})
    got = got if isinstance(got, tuple) else (got,)
    exp = exp if isinstance(exp, tuple) else (exp,)
    for (gk, gv), (ek, ev) in zip(got, exp):
        np.testing.assert_array_equal(gk, ek)
        np.testing.assert_array_equal(gv, ev)


@pytest.mark.parametrize("factory", [_chain, _fanout, _combine])
def test_stream_equals_batch(factory):
    dag, feeds = factory()
    batch = dag(*feeds)
    stream = dag.stream(*feeds)
    batch = batch if isinstance(batch, tuple) else (batch,)
    stream = stream if isinstance(stream, tuple) else (stream,)
    for (bk, bv), (sk, sv) in zip(batch, stream):
        np.testing.assert_array_equal(bk, sk)
        np.testing.assert_array_equal(bv, sv)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `poetry run pytest tests/test_dag_identity.py -v`
Expected: `test_batch_equals_oracle` PASSES (Task 1 gives batch == oracle); `test_stream_equals_batch` FAILS — `Dag.stream` undefined.

- [ ] **Step 3: Implement `Dag.stream`**

Append to `screamer/dag.py`:

```python
    def stream(self, *args, **kwargs):
        """Drive the compiled graph live, event by event (byte-identical to __call__)."""
        from .streams import merge
        feeds = self._bind_args(args, kwargs)
        streams = [_as_stream(feeds[nm]) for nm in self._input_order]
        self._cg.reset()
        # feed the merged (key-ordered, source-tagged) events one at a time
        mk, mv, ms = merge(*streams)
        for k, v, s in zip(mk, mv, ms):
            self._cg.push_event(int(s), int(k), float(v))
        results = self._cg.drain()
        results = [(k, v.reshape(-1) if v.shape[1] == 1 else v) for (k, v) in results]
        return _align_results(results, self.align_outputs)
```

- [ ] **Step 4: Build/run + commit**

Run: `poetry run pytest tests/test_dag_identity.py -v` (all pass), then `poetry run pytest -q` (green), then:

```bash
git add screamer/dag.py tests/test_dag_identity.py
git commit -m "feat(dag): Dag.stream live driver; batch == stream == oracle"
```

---

## Self-Review

**1. Spec coverage (DAG-2b-5b):**
- `Dag` computes via the C++ engine (batch) → Task 1; align at the boundary (Python, reused) via `_align_results`. ✓
- Graph node ops C++-only: `combine_latest` rejects `func=`; only functors + `combine_latest` are graph nodes → Task 1 (streams.py + compiler-side rejection). ✓
- DAG-1 `_run` compute removed from production, relocated to `tests/_dag_oracle.py` → Task 1. ✓
- `func=` graph tests migrated to `Sub`/`Add` → Task 1. ✓
- `Dag.stream()` live driver → Task 2. ✓
- batch == stream == oracle matrix (chain/fan-out/combine) → Task 2. ✓
- Completes DAG-2. Deferred (correctly absent): `dropna`/`filter`/`split` as graph nodes (DAG-2c).

**2. Placeholder scan:** none — code shown for the align helper, oracle, `_compile_cpp`, `__call__`, and `stream`. The `_as_stream` module-level export is called out.

**3. Type consistency:** `_align_results(results, align)`, `_compile_cpp`/`_input_order`/`_cg`, `run_batch`/`push_event`/`drain`/`reset`, `_GraphBuilder.add_input`/`add_functor`/`add_combine_latest`/`compile`, and `run_oracle(dag, feeds)` are consistent across tasks. Width-1 outputs reshaped to 1-D in both `__call__` and `stream` for the DAG-1 shape contract.

---

## Follow-on (DAG complete after this)

- **DAG-2c** (future) — `dropna`/`filter`/`split` as C++ push-nodes (cardinality ops in the graph).
- **WASM/JS binding** (future) — a thin binding over the same C++ engine + `_GraphBuilder`; no compute re-implemented.
