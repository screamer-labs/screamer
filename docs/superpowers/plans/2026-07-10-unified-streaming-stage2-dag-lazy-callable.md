# Unified Streaming, Stage 2: `Dag` as a lazy callable Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make a `Dag` dispatch on input type exactly like a functor (rule A): arrays give a batch result, generators give a lazy iterator that runs the compiled graph event by event, retiring `dag.stream()`.

**Architecture:** Add a Python lazy driver that pulls events from the input iterators, k-way merges them by index (as `dag.stream` already does, but lazily), pushes each into the persistent C++ `CompiledGraph` via `push_event`, and drains the outputs that closed after each push, yielding them. This reuses the already-bound C++ primitives (`push_event`, `drain`, `flush`, `reset`) with zero new numeric C++, and is provable byte-for-byte against the existing batch `dag(...)` output.

**Tech Stack:** Python, pybind11 (existing `_CompiledGraph` bindings), numpy, pytest. No C++ build is needed unless a task says so.

## Global Constraints

- Rule A dispatch (as established in Stage 1): a computation is one callable; input type selects the mode and the output container mirrors the input. For a `Dag`: all-numpy-array feeds give a batch result (unchanged); all-generator (lazy iterator) feeds give a lazy iterator out.
- `batch == lazy`: the lazy path must produce the same values in the same order as the batch `dag(...)` path, which is the oracle. This extends the existing `batch == stream` guarantee.
- All numeric logic stays in C++ (the lazy driver only marshals events; it must not reimplement any operator math in Python).
- No em-dashes and no ` -- ` in comments, docstrings, or prose.
- Never edit version files; do not bump versions.
- `Stream` and the index-as-data change are NOT in this stage (they are a later, wide-blast-radius stage). This stage keeps the current feed and return shapes and only adds the lazy call path plus retires `dag.stream()`.

## Definitions (current state, verified by grounding)

- `screamer/dag.py`:
  - `Dag.__call__(*args, **kwargs)` (around line 331): binds feeds, builds per-input `(index, values)` via `_as_stream`, calls `self._cg.run_batch(streams)`, reshapes, and returns `self._label(_align_results(results, self.align_outputs))`.
  - `Dag.stream(*args, **kwargs)` (around line 338): resets `self._cg`, uses `merge(*val_arrays, index=idx_arrays)` to order the feeds into one index-sorted event sequence, loops `self._cg.push_event(int(src), int(k), float(v))`, then `flush()`, then `drain()`, then the same reshape + `_align_results`. Returns the same result as `__call__`.
  - `_LiveDag` (around line 148): a push session over the same `_cg` with `.push/.advance/.flush/.result`. Leave it alone in this stage.
  - `_bind_args`, `_input_order`, `_as_stream`, `_align_results`, `self._label`, `self.align_outputs` are the helpers `__call__`/`stream` use.
- `include/screamer/dag/compiled_graph.h` (all bound on `_CompiledGraph` in `bindings/bindings_dag.cpp`): `reset()`, `push_event(input_idx, index, value)` (routes one width-1 event, no reset), `flush()` (emits trailing buckets at end of input), `drain()` (returns output buffers accumulated since the last drain and clears them in place, so calling it after every push is correct and cheap), `run_batch(...)`.
- `screamer/streams.py` `merge(*values, index=...)` returns `(merged_vals, merged_sources, merged_index)`, a k-way index-ordered merge used by `dag.stream`.

## File Structure

- `screamer/dag.py` (modify): add `_LazyDag` (the lazy pull iterator) and route `Dag.__call__` to it when feeds are lazy iterators; delete `Dag.stream`.
- `tests/test_dag_lazy.py` (create): the `batch == lazy` acceptance tests.
- Any test that calls `dag.stream(...)` (modify): switch to `dag(...)` on iterators, or to batch, per the retirement.

## Interfaces

- Produces: `Dag.__call__` returns a lazy iterator when every feed is a lazy iterator (a generator or `iter(...)`), and the current batch result otherwise. The lazy iterator yields output rows incrementally; materializing it (`list(...)`) yields a sequence equal to the batch result's rows.
- Consumes: the bound `_CompiledGraph` methods `reset`, `push_event`, `flush`, `drain`.

---

### Task 1: `_LazyDag` pull iterator over `push_event` + `drain`

**Files:**
- Modify: `screamer/dag.py` (add `_LazyDag`, a module-level class near `_LiveDag`)
- Test: `tests/test_dag_lazy.py`

**Interfaces:**
- Produces: `_LazyDag(dag, feeds_by_name)` -> a Python iterator yielding output rows; on exhaustion of all input iterators it flushes and yields the trailing rows, then stops.
- Consumes: `dag._cg` (`reset`/`push_event`/`flush`/`drain`), `dag._input_order`, `dag._label`, `_align_results`.

- [ ] **Step 1: Write the failing test** (tests/test_dag_lazy.py)

```python
import numpy as np
from screamer import Input, Dag, RollingMean, Sub
from screamer.streams import combine_latest


def _spread_dag():
    a, b = Input("a"), Input("b")
    return Dag(inputs=[a, b], outputs=[RollingMean(3)(Sub()(combine_latest(a, b)))]), a, b


def test_dag_lazy_equals_batch_single_output():
    dag, a, b = _spread_dag()
    va, ia = np.array([10.0, 20.0, 30.0, 40.0, 50.0]), np.array([1, 2, 3, 4, 5])
    vb, ib = np.array([1.0, 2.0, 3.0, 4.0, 5.0]),     np.array([1, 2, 3, 4, 5])

    batch_v, batch_i = dag((va, ia), (vb, ib))          # arrays -> batch

    # generators (lazy iterators) of (value, index) events -> lazy iterator out
    ga = ((float(v), int(k)) for v, k in zip(va, ia))
    gb = ((float(v), int(k)) for v, k in zip(vb, ib))
    out = dag(ga, gb)
    assert hasattr(out, "__next__") and not isinstance(out, tuple)
    rows = list(out)                                     # [(value, index), ...]
    got_v = np.array([r[0] for r in rows])
    got_i = np.array([r[1] for r in rows])
    np.testing.assert_allclose(got_v, np.asarray(batch_v).reshape(-1), equal_nan=True)
    np.testing.assert_array_equal(got_i, np.asarray(batch_i).reshape(-1))
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_dag_lazy.py::test_dag_lazy_equals_batch_single_output -x -q`
Expected: FAIL (today passing generators to `dag(...)` does not return a lazy iterator; `_as_stream` will raise or mis-handle a generator).

- [ ] **Step 3: Implement `_LazyDag`** in `screamer/dag.py` (add near `_LiveDag`). It drives the persistent engine event by event and yields per output event. Single-output first; the class is written to also carry multi-output via `_align_results` in Task 2.

```python
class _LazyDag:
    """Lazy pull driver: run the compiled graph event by event over input iterators.

    Each feed is an iterator of (value, index) events. Events are merged by index
    (as-of, ascending) across inputs, pushed one at a time, and the outputs that
    closed after each push are yielded. On exhaustion the graph is flushed and the
    trailing rows are yielded. Values match the batch dag(...) result (the oracle).
    """
    def __init__(self, dag, feeds):
        self._dag = dag
        self._cg = dag._cg
        self._cg.reset()
        # one (index, value) iterator per input, in signature order
        self._iters = [iter(feeds[nm]) for nm in dag._input_order]
        self._heads = []                # current (index, value) per input, or None
        for it in self._iters:
            self._heads.append(self._pull(it))
        self._pending = []              # buffered output rows (value, index) not yet yielded
        self._done = False

    @staticmethod
    def _pull(it):
        try:
            v, k = next(it)
            return (int(k), float(v))
        except StopIteration:
            return None

    def __iter__(self):
        return self

    def _drain_rows(self):
        # Convert one drain() into (value, index) rows for a single output.
        for (idx_arr, val_arr) in self._cg.drain():
            for k, row in zip(idx_arr, val_arr):
                yield (row[0] if getattr(row, "shape", None) and row.shape[0] == 1 else row, int(k))

    def __next__(self):
        while True:
            if self._pending:
                return self._pending.pop(0)
            if self._done:
                raise StopIteration
            # pick the input with the smallest next index (as-of merge)
            nxt = min((h[0] for h in self._heads if h is not None), default=None)
            if nxt is None:
                # all inputs exhausted: flush trailing, then stop
                self._cg.flush()
                self._pending.extend(self._drain_rows())
                self._done = True
                continue
            # push every input whose head is at this index (same-index coalescing)
            for i, h in enumerate(self._heads):
                if h is not None and h[0] == nxt:
                    self._cg.push_event(i, nxt, h[1])
                    self._heads[i] = self._pull(self._iters[i])
            self._pending.extend(self._drain_rows())
```

Note on `_drain_rows`: `drain()` returns a list of `(index_array, values_2d)` per output. For a single-output dag this is one pair; each row of `values_2d` has width 1, so yield its scalar with the index. Task 2 generalizes to multi-output. Keep the exact reshape convention `__call__` uses (`v.reshape(-1) if v.shape[1] == 1`).

- [ ] **Step 4: Route `Dag.__call__` to `_LazyDag` for lazy feeds.** In `Dag.__call__`, before the batch path, detect the lazy case and dispatch:

```python
    def __call__(self, *args, **kwargs):
        feeds = self._bind_args(args, kwargs)
        if self._all_lazy(feeds):
            return _LazyDag(self, feeds)
        streams = [_as_stream(feeds[nm]) for nm in self._input_order]
        results = self._cg.run_batch(streams)
        results = [(k, v.reshape(-1) if v.shape[1] == 1 else v) for (k, v) in results]
        return self._label(_align_results(results, self.align_outputs))

    @staticmethod
    def _all_lazy(feeds):
        # A feed is lazy iff it is an iterator (has __next__) and is not a
        # list/tuple/ndarray/Stream (those are concrete/batch, per rule A).
        import numpy as _np
        def lazy(x):
            return (hasattr(x, "__next__")
                    and not isinstance(x, (list, tuple, _np.ndarray)))
        return len(feeds) > 0 and all(lazy(v) for v in feeds.values())
```

- [ ] **Step 5: Build/run**

Run: `python -m pytest tests/test_dag_lazy.py -x -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add screamer/dag.py tests/test_dag_lazy.py
git commit -m "feat(dag): dag(generators) returns a lazy iterator over push_event+drain"
```

---

### Task 2: multi-output lazy alignment and laziness proof

**Files:**
- Modify: `screamer/dag.py` (`_LazyDag._drain_rows` for multi-output)
- Test: `tests/test_dag_lazy.py`

**Interfaces:**
- Consumes: `_LazyDag` from Task 1; `_align_results`.
- Produces: `_LazyDag` yields correct rows for a multi-output `Dag` (align_outputs default), and is provably lazy (consumes input one event at a time).

- [ ] **Step 1: Write the failing tests** (append to tests/test_dag_lazy.py)

```python
def test_dag_lazy_is_lazy():
    # The driver must pull input events one at a time, not eagerly.
    from screamer import Input, Dag, CumSum
    pulled = []
    def spy(vals):
        for i, v in enumerate(vals):
            pulled.append(v)
            yield (v, i)
    x = Input("x")
    dag = Dag(inputs=[x], outputs=[CumSum()(x)])
    it = dag(spy([1.0, 2.0, 3.0]))
    assert pulled == []            # nothing consumed before first next()
    first = next(it)
    assert pulled == [1.0]         # exactly one input event consumed
    assert first == (1.0, 0)


def test_dag_lazy_equals_batch_multi_output():
    import numpy as np
    from screamer import Input, Dag, Sub, RollingMean
    from screamer.streams import combine_latest
    a, b = Input("a"), Input("b")
    spread = Sub()(combine_latest(a, b))
    dag = Dag(inputs=[a, b], outputs=[spread, RollingMean(2)(spread)])  # 2 outputs
    va, ia = np.array([10.0, 20.0, 30.0]), np.array([1, 2, 3])
    vb, ib = np.array([1.0, 2.0, 3.0]),   np.array([1, 2, 3])
    batch = dag((va, ia), (vb, ib))                    # tuple of (values, index) pairs
    ga = ((float(v), int(k)) for v, k in zip(va, ia))
    gb = ((float(v), int(k)) for v, k in zip(vb, ib))
    rows = list(dag(ga, gb))                            # rows of (col0, col1, index) or similar
    # one row per output index; compare column by column against batch
    assert len(rows) == len(np.asarray(batch[0][0]).reshape(-1))
```

- [ ] **Step 2: Run to verify the multi-output case fails**

Run: `python -m pytest tests/test_dag_lazy.py -x -q`
Expected: `test_dag_lazy_is_lazy` passes (Task 1 driver is already lazy); `test_dag_lazy_equals_batch_multi_output` FAILS or errors because `_drain_rows` only handles a single output.

- [ ] **Step 3: Generalize `_drain_rows` for multi-output.** For a multi-output dag with `align_outputs=True`, each drained event across the M outputs at the same index becomes one row `(col0, col1, ..., index)`. Implement it by draining all M output buffers, grouping their rows by index, and yielding `(values..., index)` in index order. Keep single-output behavior (a bare scalar plus index) unchanged. Reuse the same per-output reshape convention and, where the batch path calls `_align_results`, mirror its as-of coalescing so the lazy rows equal the batch rows.

```python
    def _drain_rows(self):
        drained = self._cg.drain()                       # list of (idx_arr, vals_2d) per output
        if len(drained) == 1:
            idx_arr, vals = drained[0]
            for k, row in zip(idx_arr, vals):
                yield (float(row[0]) if vals.shape[1] == 1 else tuple(map(float, row)), int(k))
            return
        # multi-output: group by index across outputs, one row per distinct index
        import numpy as _np
        by_index = {}
        for out_pos, (idx_arr, vals) in enumerate(drained):
            for k, row in zip(idx_arr, vals):
                cols = by_index.setdefault(int(k), [None] * len(drained))
                cols[out_pos] = float(row[0]) if vals.shape[1] == 1 else tuple(map(float, row))
        for k in sorted(by_index):
            yield tuple(by_index[k]) + (k,)
```

(If the exact multi-output row shape or the as-of alignment does not match the batch `_align_results` output, adjust so `list(dag(gens))` equals the batch result row for row; the test is the contract.)

- [ ] **Step 4: Build/run**

Run: `python -m pytest tests/test_dag_lazy.py -x -q`
Expected: PASS (all three tests).

- [ ] **Step 5: Commit**

```bash
git add screamer/dag.py tests/test_dag_lazy.py
git commit -m "feat(dag): multi-output lazy alignment; prove batch == lazy"
```

---

### Task 3: retire `Dag.stream`

**Files:**
- Modify: `screamer/dag.py` (delete `Dag.stream`)
- Modify: any test that calls `dag.stream(...)` (grep and update)
- Test: full suite

**Interfaces:**
- Consumes: `dag(...)` lazy path (replaces `dag.stream`).

- [ ] **Step 1: Grep for `dag.stream` / `.stream(` uses**

Run: `grep -rn "\.stream(" screamer/ tests/ docs/ | grep -iv "def stream\|streams\.\|streaming"`
Expected: a list of call sites (tests and possibly docs). Each is either replaced by feeding iterators to `dag(...)`, or by the batch `dag(...)` when it was only asserting `stream == batch`.

- [ ] **Step 2: Delete `Dag.stream`** in `screamer/dag.py` (the method around line 338). Update each grepped test: a `dag.stream(feed_a, feed_b)` call that fed arrays and asserted equality with `dag(...)` becomes a `dag(...)` on `(value, index)` generators of those arrays, asserting equality with the batch `dag(...)`. Do not weaken assertions.

- [ ] **Step 3: Run the full suite**

Run: `python -m pytest -q`
Expected: PASS. Baseline is the current `main` suite count after Stage 1 (`3924 passed`, plus the 2 pre-existing `test_oscillators_hlc.py::TestBOP` failures that are NOT related, and 2 skips). This stage must add zero new failures; the new `tests/test_dag_lazy.py` is additive.

- [ ] **Step 4: Commit**

```bash
git add screamer/dag.py tests/
git commit -m "refactor(dag): retire dag.stream() in favor of dag(iterables)"
```

---

## Self-review notes

- **Spec slice covered:** rule A dispatch for `Dag` (generators give a lazy iterator, arrays give batch) and the retirement of `dag.stream()`. The lazy driver reuses the C++ `push_event`+`drain` engine, so no operator math moves to Python. `batch == lazy` is the enforced oracle.
- **Not in this stage (later stages):** `dag.live()`/`advance()` retirement and the push form; making the stream operators (`resample` etc.) themselves lazy callables and retiring `*_iter`; retiring `Stream`; index-as-data and the `resample` `freq=` re-signature.
- **Risk to watch during execution:** the exact row shape of the multi-output lazy path must equal the batch `_align_results` output; the test is the contract, and Task 2 says to adjust the shaping to match. If aligning incrementally proves subtle, an acceptable fallback for this slice is to align per drained index using the same as-of rule `_align_results` uses, since outputs at a shared index arrive together.

## Next stages (separate plans)

3. Stream operators as lazy callables: route `resample`/`dropna`/`select` streaming through their existing C++ DAG nodes via the push+drain driver, retire `resample_iter`/`dropna_iter`/`select_iter` (and the Python `_ResampleAccum`); handle `filter` (Python predicate, no C++ node) and `merge`/`combine_latest` (existing C++ pullers) explicitly.
4. `resample` `freq=` re-signature, tuple output, `(index, NaN)` heartbeats; retire `advance()`/`dag.live()`.
5. Retire `Stream` (wide blast radius) and index-as-data.
6. Docs and notebooks onto the one surface.
