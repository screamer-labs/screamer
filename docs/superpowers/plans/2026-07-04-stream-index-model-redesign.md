# Stream / index model redesign - Implementation Plan (revised)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Implement the finalized redesign: a stream is values with an optional index; combinators (stream operators) are polymorphic on input type (raw arrays / `Stream` / `Node`) with mirrored returns; `combine_latest` emits one row per distinct index (same-index events coalesce); no-index means aligned clocks (lockstep, equal length); `key`->`index` and `combinator`->`stream operator` everywhere; `resample` `width`->`every`.

**Reference:** spec `docs/superpowers/specs/2026-07-04-stream-index-model-redesign.md` (see the Vocabulary table and Decisions log).

**Done already on this branch:** Task 1 - the `Stream` type + `_regime`/`_to_streams`/`_adapt` dispatch helpers.

## Global Constraints

- **Vocabulary is binding** (spec's table): value(s), **index** (never "key"), **stream** (never series/feed), **event** (streaming only), **row** (2-D output only), **positional** (index=None), **stream operator** (never "combinator"), functor. Applies to code identifiers, docstrings, and docs.
- **Identity preserved and strengthened:** batch == stream == graph == eager-oracle. The redesign must keep every existing causal/identity guarantee; `combine_latest` gains coalescing which makes the modes converge.
- **Values-first, index optional.** `index=None` -> positional, no allocation on return. Raw return always `(values, index)`; `Stream` -> `Stream`; `Node` -> `Node`.
- **No-index alignment ops require equal length** (`combine_latest`); unequal is a clear error.
- Build gotcha: after any C++ change run `make install-dev` (not `make build`). No version-file / `screamer/__init__.py` edits.

---

### Task A: Naming sweep (`key`->`index`, `combinator`->`stream operator`, `series`->`streams`)

Pure rename, zero behavior change, full suite stays green. Do this first so later logic changes never mix with rename diffs.

**Files:** C++ under `include/screamer/dag/`, `include/screamer/streams/`, `bindings/`, `src/`; Python `screamer/streams.py`, `screamer/dag.py`; tests referencing these identifiers.

- [ ] **Step 1: C++ `Key` -> `Index`.** Rename the `template <class Key>` parameter to `template <class Index>` and every `Key`/`key` identifier to `Index`/`index` across `include/screamer/dag/*.h` (frame.h, functor_node.h, combine_latest_node.h, dropna_node.h, select_node.h, resample_node.h, broadcast.h, graph.h, compiled_graph.h, driver.h, collector.h) and `include/screamer/streams/*.h`. This is mechanical (the template arg and local variable names); the instantiation type `std::int64_t` is unchanged. Grep to confirm no `Key`/`key` remains in these headers.
- [ ] **Step 2: `combinator` -> operator identifiers.** In `screamer/dag.py` and `screamer/streams.py`: `make_combinator_node` -> `make_operator_node`; the op-tag string `"combinator"` -> `"operator"`; any `combinator` in comments/docstrings -> "stream operator". In bindings, `add_combinator`-style names if present. Update all call sites and imports.
- [ ] **Step 3: `series` -> `streams`.** `_normalize_series` -> `_normalize_streams`; `*series` params -> `*streams`; `who="combine_latest"`-style strings unaffected. Update `_merge_events` and any `series` locals.
- [ ] **Step 4: Build + full suite.** `make install-dev && poetry run pytest -q`. Expected: identical pass count to the pre-rename base (3209 passed), because nothing changed but names. Grep the C++ headers and the two Python modules for a stray `\bkey\b`/`\bKey\b`/`combinator`/`\bseries\b` and fix any missed.
- [ ] **Step 5: Commit** `refactor: rename key->index, combinator->stream operator, series->streams (no behavior change)`.

---

### Task B: `combine_latest` - values-first, polymorphic, **coalescing**, aligned-clocks rule

**Files:** `screamer/streams.py`; `include/screamer/dag/combine_latest_node.h` (coalesce + flush); test `tests/test_streams_combine_latest.py` (create).

**Interfaces:** `combine_latest(*values, index=None, emit="when_all", func=None)` -> `(values, index)` / `Stream` / `Node`, emitting one row per distinct index. `combine_latest_iter(*values, index=None, emit="when_all")` yields coalesced `(row, index)` events.

- [ ] **Step 1: Write the failing test** `tests/test_streams_combine_latest.py`:

```python
import numpy as np
import pytest
from screamer.streams import combine_latest, combine_latest_iter, Stream


def test_positional_lockstep_equal_length():
    v, idx = combine_latest(np.array([10.0, 20.0, 40.0]), np.array([1.0, 3.0, 4.0]))
    assert idx is None
    assert v.shape == (3, 2)                       # lockstep, one row per position
    np.testing.assert_array_equal(v[:, 0], [10.0, 20.0, 40.0])


def test_positional_unequal_length_raises():
    with pytest.raises(ValueError, match="length"):
        combine_latest(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0]))


def test_indexed_coalesces_same_key_rows():
    a = np.array([10.0, 20.0, 40.0]); ta = np.array([1, 2, 4])
    b = np.array([1.0, 3.0, 4.0]);    tb = np.array([1, 3, 4])
    v, idx = combine_latest(a, b, index=[ta, tb])
    np.testing.assert_array_equal(idx, [1, 2, 3, 4])          # NOT [1,2,3,4,4]
    np.testing.assert_array_equal(v, [[10, 1], [20, 1], [20, 3], [40, 4]])


def test_stream_in_stream_out():
    a = Stream(np.array([10.0, 20.0]), np.array([1, 2]))
    b = Stream(np.array([1.0, 2.0]), np.array([1, 2]))
    out = combine_latest(a, b)
    assert isinstance(out, Stream)
    np.testing.assert_array_equal(out.index, [1, 2])


def test_mixed_positional_and_indexed_raises():
    with pytest.raises(ValueError, match="positional"):
        combine_latest(Stream(np.array([1.0, 2.0]), np.array([1, 2])),
                       Stream(np.array([3.0, 4.0])))


def test_node_in_node_out():
    from screamer import Input
    from screamer.dag import is_node
    assert is_node(combine_latest(Input("a"), Input("b")))


def test_iter_coalesces_and_yields_row_index():
    a = Stream(np.array([10.0, 20.0]), np.array([1, 2]))
    b = Stream(np.array([1.0, 2.0]), np.array([1, 2]))
    events = list(combine_latest_iter(a, b))
    idxs = [i for _, i in events]
    assert idxs == [1, 2]                            # one event per distinct index
    np.testing.assert_array_equal(np.asarray(events[0][0]), [10.0, 1.0])
```

- [ ] **Step 2: Run to verify failures.**

- [ ] **Step 3a: Rewrite `combine_latest` (eager coalescing).** Add a collapse helper and the dispatch wrapper:

```python
def _collapse_last_per_index(index, values):
    """Keep the last row of each run of equal index (one row per distinct index).
    index must be non-decreasing (aligner emits in index order)."""
    n = len(index)
    if n == 0:
        return index, values
    keep = np.empty(n, dtype=bool)
    keep[:-1] = index[:-1] != index[1:]      # last of each equal-index run
    keep[-1] = True
    return index[keep], values[keep]


def _streams_to_keyed(streams, who):
    """(kind, index_list, vals_list, positional). Uniform positional or indexed."""
    indexed = [s.index is not None for s in streams]
    if any(indexed) and not all(indexed):
        raise ValueError(
            f"{who}: cannot align positional and indexed streams; give every "
            "stream an index, or none")
    vals = [np.ascontiguousarray(s.values, dtype=np.float64) for s in streams]
    if not any(indexed):
        lens = {len(s) for s in streams}
        if len(lens) != 1:
            raise ValueError(
                f"{who}: streams have no index, so they are assumed aligned - "
                "lengths must match, or provide an index to align different clocks")
        idx = [np.arange(len(streams[0]), dtype=np.int64) for _ in streams]
        return "i64", idx, vals, True
    kind, idx, _ = _normalize_streams([(s.index, s.values) for s in streams], who)
    return kind, idx, vals, False


def combine_latest(*values, index=None, emit="when_all", func=None):
    """As-of latest-value join of N streams: one row per distinct index (same-index
    events coalesce). See the Streams docs for the model."""
    if emit not in ("when_all", "on_any"):
        raise ValueError('combine_latest: emit must be "when_all" or "on_any"')
    if any(is_node(v) for v in values):
        if func is not None:
            raise ValueError(
                "combine_latest(func=...) is not supported in a DAG graph "
                "(graph ops are C++-only); apply a functor to the aligned output, "
                "e.g. Sub()(combine_latest(a, b))")
        return make_operator_node(combine_latest, values, {"emit": emit, "func": None})
    regime = _regime(values)
    streams = _to_streams(values, index)
    kind, idx, vals, positional = _streams_to_keyed(streams, "combine_latest")
    fn = _b._combine_latest_f64 if kind == "f64" else _b._combine_latest_i64
    out_index, aligned = fn(idx, vals, emit == "when_all")
    out_index, aligned = _collapse_last_per_index(out_index, aligned)   # coalesce
    result_index = None if positional else out_index
    if func is not None:
        aligned = np.array([func(*row) for row in aligned], dtype=np.float64)
    return _adapt(regime, aligned, result_index)
```

- [ ] **Step 3b: Rewrite `combine_latest_iter` (streaming coalescing).** Buffer the current index's row and emit it when the index advances (and at end):

```python
def combine_latest_iter(*values, index=None, emit="when_all"):
    """Yield coalesced (row, index) events: one per distinct index."""
    if emit not in ("when_all", "on_any"):
        raise ValueError('combine_latest: emit must be "when_all" or "on_any"')
    streams = _to_streams(values, index)
    kind, idx, vals, positional = _streams_to_keyed(streams, "combine_latest")
    cls = _b._CombineLatestPuller_f64 if kind == "f64" else _b._CombineLatestPuller_i64
    puller = cls(idx, vals, emit == "when_all")
    cur_index, cur_row = None, None
    while True:
        event = puller.next()
        if event is None:
            break
        ev_index, ev_row = event
        if cur_index is not None and ev_index != cur_index:
            yield cur_row, (None if positional else cur_index)     # index advanced -> emit last
        cur_index, cur_row = ev_index, ev_row
    if cur_index is not None:
        yield cur_row, (None if positional else cur_index)         # flush final index
```
(Confirm the puller's `next()` shape - `(index, row)` - against the live code and adapt the unpack.)

- [ ] **Step 3c: Coalesce in the graph combine node.** In `include/screamer/dag/combine_latest_node.h` (now `Index`-templated after Task A), change `CombineLatestNode` to emit one frame per distinct index: buffer the latest aligned row for the current index and push it downstream only when an event with a larger index arrives; add `flush()` that pushes the buffered final index's row then forwards `downstream.flush()`; `reset()` clears the buffer. This mirrors the resample emit-on-boundary + flush. `Dag.stream` already calls `flush()` before drain (from the resample work), so no dag.py change is needed for the trigger. Register the node's flush behavior so `run_batch` (which already flushes at end) coalesces too.

- [ ] **Step 4: Build + run** `make install-dev && poetry run pytest tests/test_streams_combine_latest.py tests/test_dag_identity.py -q`. The graph identity (`_combine`, `_fanout`) must now show the coalesced cardinality; update those oracle expectations if they encoded the old per-event counts (the oracle `tests/_dag_oracle.py` and `test_dag_identity.py` may need the coalesced numbers - align them to the new semantics, do not weaken the batch==stream==graph assertion).

- [ ] **Step 5: Commit** `feat(streams): combine_latest one-row-per-index (coalesce) + values-first polymorphic`.

---

### Task C: `dropna`, `filter`, `select` (+ `_iter`)

**Interfaces:** `dropna(values, index=None, how="any")`, `filter(values, predicate, index=None)`, `select(values, columns, index=None)` -> `(values, index)` / `Stream` / `Node` (filter eager-only). `_iter` twins yield `(value, index)` events.

Single-stream shape operators - no alignment/coalescing/mixing. Apply the wrapper: `Node` -> `make_operator_node`; else `_regime` on the single input -> one `Stream` -> existing masking/projection on `.values`, threading `.index` through the same mask/pass-through (or `None`) -> `_adapt`.

- [ ] **Step 1:** Update `tests/test_streams_shape.py`, `tests/test_streams_select.py`, and the graph tests `tests/test_dag_dropna.py`/`tests/test_dag_select.py` to the new signatures; add raw/`Stream`/`Node` mirroring, `index is None` positional, index masked-with-values for dropna, and the `filter`-on-`Node` rejection.
- [ ] **Step 2:** Run to verify failures.
- [ ] **Step 3:** Rewrite the six functions per the wrapper; keep the numpy logic; thread `index`.
- [ ] **Step 4:** `poetry run pytest tests/test_streams_shape.py tests/test_streams_select.py tests/test_dag_dropna.py tests/test_dag_select.py -q`.
- [ ] **Step 5:** Commit `feat(streams): dropna/filter/select values-first polymorphic`.

---

### Task D: `resample` (+ `_iter`), `width`->`every`

**Interfaces:** `resample(values, index=None, *, every=None, count=None, agg="last", origin=0, label="left")` -> `(values, index)` / `Stream` / `Node`; returned `index` is the bar labels. `dag.py` operator dispatch reads `kwargs["every"]`; the C++/binding keep their internal `width` argument name.

- [ ] **Step 1:** Update `tests/test_streams_resample.py`, `tests/test_dag_resample.py` to `every=` and `(values, index)`; add raw/`Stream`/`Node` mirroring; keep every agg/mode/ohlc/NaN/negative-index/trailing-flush assertion.
- [ ] **Step 2:** Verify failures.
- [ ] **Step 3:** Rewrite `resample`/`resample_iter` (wrapper + `every`); update `dag.py` operator dispatch to read `every`.
- [ ] **Step 4:** `poetry run pytest tests/test_streams_resample.py tests/test_dag_resample.py tests/test_dag_identity.py -q`.
- [ ] **Step 5:** Commit `feat(streams): resample values-first + rename width to every`.

---

### Task E: `merge`, `split`, `pace` + keyless arrival index

**Interfaces:** `merge(*values, index=None) -> (values, sources, index)` (never coalesces; positional/indexed mixing raises); `merge_iter` yields `(value, index, source)`; `split(values, sources, index=None, n=None) -> list[(values, index)]`; `pace(*values, index=None, speed=1.0, sleep=None)` yields `(value, index, source)`. Keyless live input is stamped with a monotonic arrival index at the streaming boundary (so `combine_latest_iter`/`pace` over keyless sources emit per arrival).

- [ ] **Step 1:** Update `tests/test_streams_merge.py`, `tests/test_streams_identity.py` to the new signatures + event shapes; add positional (`index is None`) and a keyless-arrival test.
- [ ] **Step 2:** Verify failures.
- [ ] **Step 3:** Rewrite `merge`/`merge_iter`/`split`/`pace`; add arrival-index stamping for keyless live sources.
- [ ] **Step 4:** `poetry run pytest tests/test_streams_merge.py tests/test_streams_identity.py -q`.
- [ ] **Step 5:** Commit `feat(streams): merge/split/pace values-first; keyless arrival index`.

---

### Task F: DAG boundary

**Interfaces:** `Dag` feeds accept bare arrays / `Stream`s / `(values, index)`; `_as_stream` normalizes to the engine's `(index, values)`; `Dag(...)` returns `(values, index)` shape; `_align_results` uses `combine_latest(*values, index=[...])`.

- [ ] **Step 1:** Update `tests/test_dag_identity.py` + `tests/test_dag_*.py` to `values, index = dag(...)`; add a `Stream`-fed `Dag`; keep batch==stream assertions.
- [ ] **Step 2:** Verify failures.
- [ ] **Step 3:** Update `_as_stream`, `Dag.__call__`/`stream`/`_align_results`; grep `screamer/` for any residual user-facing `key` and fix.
- [ ] **Step 4:** `poetry run pytest tests/test_dag_identity.py tests/test_dag_dropna.py tests/test_dag_select.py tests/test_dag_resample.py -q`.
- [ ] **Step 5:** Commit `feat(dag): Stream feeds; (values, index) returns`.

---

### Task G: Documentation

- [ ] **Step 1:** Rewrite `docs/multistream.md` to the model + Vocabulary (a stream is values; index is an optional ordering coordinate; `combine_latest` = one row per distinct index; no-index=aligned-clocks; keyless-live=arrival index). Dash-free.
- [ ] **Step 2:** Update `docs/functions_streams/*.md` + `functions_dag/Dag.md` examples to new signatures (values-first, `index=`, `every=`, `(values, index)`, `Stream` shown once); one focused build-executed result each.
- [ ] **Step 3:** Update notebooks 07-10 to the new signatures/vocabulary; `poetry run pytest --nbmake docs/notebooks/ -q` (all 10 green).
- [ ] **Step 4:** `make docs` clean (baseline warnings only); no "key"/"combinator"/dict framing remains on the stream pages.
- [ ] **Step 5:** Commit `docs: reframe streams around values + index; vocabulary sweep`.

---

### Task H: Final verification

- [ ] **Step 1:** `poetry run pytest -q` full green.
- [ ] **Step 2:** `poetry run pytest --nbmake docs/notebooks/ -q` all green.
- [ ] **Step 3:** Spot-check the three regimes for `combine_latest` and `resample` (raw -> `(values, index)` with `index is None` positional; `Stream` -> `Stream`; `Node` -> graph), the coalesced cardinality (distinct-index count; equal-length-positional -> N; unequal-positional -> error), and batch == stream == graph.
- [ ] **Step 4:** Grep the whole tree for residual `\bkey\b`/`combinator`/`\bseries\b` on the public surface; confirm none. Commit any fixups; branch ready for whole-branch review.

---

## Self-Review

**Coverage:** naming sweep incl. C++ `Key`->`Index` (A); `combine_latest` coalescing + aligned-clocks rule + polymorphic (B); shape ops (C); resample + `every` (D); merge/split/pace + arrival index (E); DAG boundary (F); docs (G); final incl. cardinality + coalescing + vocabulary checks (H). Matches spec Decisions log 1-12.

**Risk notes:** A is a pure rename verified by an unchanged pass count. B is the semantic change: the collapse helper is a simple last-per-run mask (batch), a small emit-on-advance generator (iter), and an emit-on-index-advance + flush on the combine node (graph, reusing resample flush) - and it is what makes batch==stream==graph hold. The graph identity oracle numbers change to the coalesced cardinality; align them to the new semantics without weakening the cross-mode equality.
