# Stream / index model redesign - Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Redesign the multi-stream interface so a stream is values with an optional index; combinators dispatch on input type (raw arrays / `Stream` / `Node`) and mirror it on return; raw return is always `(values, index)` with `index=None` for positional (no allocation); rename `key(s)` -> `index` everywhere user-facing. The engine and all causal/identity guarantees are unchanged.

**Architecture:** A new `Stream` value type plus a small dispatch layer that normalizes every input regime to `Stream`, runs the *existing* combinator cores unchanged, and adapts the output back to match the input regime. Combinators are re-signatured values-first with an optional `index=`.

**Tech Stack:** Python 3.11, numpy, pandas (interop), pytest. No C++ changes.

**Reference:** spec `docs/superpowers/specs/2026-07-04-stream-index-model-redesign.md`.

## Global Constraints

- **Engine untouched.** No changes under `include/`, `bindings/`, or `src/`. The C++ aligner/resampler/DAG stay as-is; this is a Python interface layer over them.
- **Identity preserved.** batch == streaming == oracle must still hold. Existing behavior is unchanged; only signatures and in/out wrapping change. Update existing tests to the new signatures; do not weaken their assertions.
- **Values-first, index optional.** Every public signature leads with values; `index=` is a keyword. `index=None` means positional and allocates nothing on return.
- **`key` -> `index`** across the public surface (params, returns, docstrings). Internal C++ `Key` and internal `key` variables stay.
- **Raw return is always `(values, index)`** (2-tuple, values first, `index is None` for positional). `Stream` in -> `Stream` out. `Node` in -> `Node` out.
- **Alignment ops (combine_latest, merge) require uniform indexing:** all inputs positional, or all indexed. Mixing positional and indexed inputs raises a clear error.
- No version-file / `screamer/__init__.py` edits; run `poetry run pytest` (pure Python, no build needed).

---

### Task 1: `Stream` type + dispatch layer

**Files:**
- Modify: `screamer/streams.py` (add `Stream` + dispatch helpers near the top, after imports)
- Test: `tests/test_stream_type.py` (create)

**Interfaces:**
- Produces: `Stream(values, index=None)` with `.values`, `.index` (None == positional), `__len__`, `Stream.from_pandas`, `.to_pandas`. Dispatch helpers `_regime(inputs)`, `_to_streams(inputs, index)`, `_adapt(regime, values, index)`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_stream_type.py`:

```python
import numpy as np
import pytest
from screamer.streams import Stream


def test_stream_positional_index_none():
    s = Stream(np.array([1.0, 2.0, 3.0]))
    assert s.index is None
    assert len(s) == 3
    np.testing.assert_array_equal(s.values, [1.0, 2.0, 3.0])


def test_stream_with_index():
    s = Stream(np.array([1.0, 2.0]), index=np.array([10, 20]))
    np.testing.assert_array_equal(s.index, [10, 20])


def test_stream_length_mismatch_raises():
    with pytest.raises(ValueError, match="same length"):
        Stream(np.array([1.0, 2.0, 3.0]), index=np.array([10, 20]))


def test_stream_pandas_roundtrip():
    pd = pytest.importorskip("pandas")
    ser = pd.Series([1.0, 2.0, 3.0], index=[100, 200, 300])
    s = Stream.from_pandas(ser)
    np.testing.assert_array_equal(s.values, [1.0, 2.0, 3.0])
    np.testing.assert_array_equal(s.index, [100, 200, 300])
    back = s.to_pandas()
    np.testing.assert_array_equal(back.to_numpy(), [1.0, 2.0, 3.0])
    np.testing.assert_array_equal(np.asarray(back.index), [100, 200, 300])


def test_stream_to_pandas_positional_default_index():
    pd = pytest.importorskip("pandas")
    s = Stream(np.array([1.0, 2.0]))
    ser = s.to_pandas()
    np.testing.assert_array_equal(np.asarray(ser.index), [0, 1])
```

- [ ] **Step 2: Run test to verify it fails** — `poetry run pytest tests/test_stream_type.py -q` (ImportError: cannot import name 'Stream').

- [ ] **Step 3: Implement `Stream` + dispatch helpers**

In `screamer/streams.py`, after the existing imports (which include `is_node`, `make_combinator_node` from `.dag`), add:

```python
class Stream:
    """A sequence of values with an optional ordering index.

    values : np.ndarray, shape (T,) or (T, N).
    index  : np.ndarray of length T, or None. None means positional (row-number /
             arrival order) and stores nothing. The index is an ordering
             coordinate (timestamp, tick counter, ...), never a lookup key.
    """
    __slots__ = ("values", "index")

    def __init__(self, values, index=None):
        self.values = np.asarray(values)
        self.index = None if index is None else np.asarray(index)
        if self.index is not None and len(self.index) != len(self.values):
            raise ValueError("Stream: index and values must have the same length")

    def __len__(self):
        return len(self.values)

    def __repr__(self):
        kind = "positional" if self.index is None else f"index={self.index!r}"
        return f"Stream({self.values!r}, {kind})"

    @classmethod
    def from_pandas(cls, obj):
        """Build a Stream from a pandas Series or DataFrame (data -> values,
        pandas index -> index)."""
        return cls(obj.to_numpy(), np.asarray(obj.index))

    def to_pandas(self):
        """Return a pandas Series (1-D values) or DataFrame (2-D). A positional
        stream gets pandas' default RangeIndex."""
        import pandas as pd
        if self.values.ndim == 1:
            return pd.Series(self.values, index=self.index)
        return pd.DataFrame(self.values, index=self.index)


def _regime(inputs):
    """Classify a combinator's inputs: 'graph' if any is a Node, 'stream' if any
    is a Stream, else 'raw'."""
    if any(is_node(x) for x in inputs):
        return "graph"
    if any(isinstance(x, Stream) for x in inputs):
        return "stream"
    return "raw"


def _to_streams(inputs, index):
    """Normalize each input to a Stream. `index` is None (all positional) or a
    list aligned with inputs (per-stream index array or None)."""
    if index is not None and len(index) != len(inputs):
        raise ValueError("index list length must match the number of streams")
    out = []
    for i, x in enumerate(inputs):
        if isinstance(x, Stream):
            out.append(x)
        else:
            out.append(Stream(x, None if index is None else index[i]))
    return out


def _adapt(regime, values, index):
    """Shape a combinator result to match the input regime: Stream in -> Stream
    out; raw -> (values, index) with index None for positional."""
    if regime == "stream":
        return Stream(values, index)
    return values, index
```

- [ ] **Step 4: Run test to verify it passes** — `poetry run pytest tests/test_stream_type.py -q` (5 passed).

- [ ] **Step 5: Commit**

```bash
git add screamer/streams.py tests/test_stream_type.py
git commit -m "feat(streams): add Stream type + input-regime dispatch helpers"
```

---

### Task 2: Migrate `combine_latest` (+ `_iter`) - establishes the pattern

**Files:**
- Modify: `screamer/streams.py` (`combine_latest`, `combine_latest_iter`, `_normalize_series`)
- Test: `tests/test_streams_combine_latest.py` (create or update)

**Interfaces:**
- Consumes: `Stream`, `_regime`, `_to_streams`, `_adapt` (Task 1).
- Produces: `combine_latest(*values, index=None, emit="when_all", func=None)` returning `(aligned_values, index)` (raw), `Stream` (Stream in), or `Node` (Node in). `combine_latest_iter(*values, index=None, emit="when_all")` yielding `(value_row, index)` per event.

**The pattern (applies to every combinator):** keep the existing C++-backed core untouched; wrap it. (1) validate params; (2) `Node` in any position -> `make_combinator_node` (graph, unchanged); (3) classify `regime`; (4) `_to_streams`; (5) decide positional vs indexed (all `index is None` -> positional: synthesize row-number keys to drive the core, output index None; all indexed -> use their indices as keys, output index = the core's returned keys; mixed -> error); (6) call the existing core; (7) `_adapt` the output.

- [ ] **Step 1: Write the failing test**

Create `tests/test_streams_combine_latest.py`:

```python
import numpy as np
import pytest
from screamer.streams import combine_latest, Stream


def test_raw_positional_returns_values_and_none_index():
    a = np.array([10.0, 11.0, 12.0])
    b = np.array([1.0, 2.0, 3.0])
    values, index = combine_latest(a, b)
    assert index is None                       # positional -> no allocation
    assert values.shape == (3, 2)
    np.testing.assert_array_equal(values[:, 0], [10.0, 11.0, 12.0])


def test_raw_indexed_returns_union_index():
    a = np.array([10.0, 11.0, 13.0]); ta = np.array([1, 2, 4])
    b = np.array([5.0, 7.0, 9.0]);    tb = np.array([1, 3, 4])
    values, index = combine_latest(a, b, index=[ta, tb])
    np.testing.assert_array_equal(index, [1, 2, 3, 4, 4])
    assert values.shape == (5, 2)


def test_stream_in_stream_out():
    a = Stream(np.array([10.0, 11.0, 13.0]), np.array([1, 2, 4]))
    b = Stream(np.array([5.0, 7.0, 9.0]), np.array([1, 3, 4]))
    out = combine_latest(a, b)
    assert isinstance(out, Stream)
    np.testing.assert_array_equal(out.index, [1, 2, 3, 4, 4])


def test_mixing_positional_and_indexed_raises():
    a = Stream(np.array([1.0, 2.0]), np.array([1, 2]))
    b = Stream(np.array([3.0, 4.0]))                  # positional
    with pytest.raises(ValueError, match="positional"):
        combine_latest(a, b)


def test_node_in_node_out():
    from screamer import Input
    from screamer.dag import is_node
    a, b = Input("a"), Input("b")
    assert is_node(combine_latest(a, b))


def test_func_reduction_raw():
    a = np.array([10.0, 20.0]); b = np.array([1.0, 2.0])
    values, index = combine_latest(a, b, func=lambda x, y: x - y)
    assert index is None
    np.testing.assert_array_equal(values, [9.0, 18.0])


def test_iter_yields_value_index_pairs():
    a = Stream(np.array([10.0, 11.0]), np.array([1, 2]))
    b = Stream(np.array([5.0, 7.0]), np.array([1, 2]))
    from screamer.streams import combine_latest_iter
    events = list(combine_latest_iter(a, b))
    # each event is (value_row, index)
    row0, idx0 = events[0]
    assert idx0 == 1
    np.testing.assert_array_equal(np.asarray(row0), [10.0, 5.0])
```

- [ ] **Step 2: Run to verify it fails** — new-signature calls fail against the old `(keys, values)`-pair signature.

- [ ] **Step 3: Rewrite `combine_latest`**

Replace the body with the dispatch wrapper. Keep `_normalize_series` for the indexed path; add a positional path that synthesizes row-number keys. Concretely:

```python
def _streams_to_keyed(streams, who):
    """Return (kind, keys_list, vals_list, out_index_is_positional). All streams
    must be uniformly positional or uniformly indexed."""
    indexed = [s.index is not None for s in streams]
    if any(indexed) and not all(indexed):
        raise ValueError(
            f"{who}: cannot align positional and indexed streams; give every "
            "stream an index, or none")
    vals = [np.ascontiguousarray(s.values, dtype=np.float64) for s in streams]
    if not any(indexed):
        # positional: synthesize int64 row-number keys to drive the core
        keys = [np.arange(len(s), dtype=np.int64) for s in streams]
        return "i64", keys, vals, True
    # indexed: reuse the existing key-dtype normalization
    kind, keys, _ = _normalize_series([(s.index, s.values) for s in streams], who)
    return kind, keys, vals, False


def combine_latest(*values, index=None, emit="when_all", func=None):
    """As-of latest-value join of N streams. See the Streams docs for the model.

    values : the input streams (value arrays, Streams, or graph Nodes).
    index  : None (all positional), or a list of per-stream index arrays.
    Returns (aligned_values, index) for raw input (index is None when positional),
    a Stream for Stream input, or a Node for graph input.
    """
    if emit not in ("when_all", "on_any"):
        raise ValueError('combine_latest: emit must be "when_all" or "on_any"')
    if any(is_node(v) for v in values):
        if func is not None:
            raise ValueError(
                "combine_latest(func=...) is not supported in a DAG graph "
                "(graph ops are C++-only); apply a C++ functor to the aligned "
                "output instead, e.g. Sub()(combine_latest(a, b))")
        return make_combinator_node(combine_latest, values, {"emit": emit, "func": None})
    regime = _regime(values)
    streams = _to_streams(values, index)
    kind, keys, vals, positional = _streams_to_keyed(streams, "combine_latest")
    fn = _b._combine_latest_f64 if kind == "f64" else _b._combine_latest_i64
    out_keys, aligned = fn(keys, vals, emit == "when_all")
    out_index = None if positional else out_keys
    if func is not None:
        aligned = np.array([func(*row) for row in aligned], dtype=np.float64)
    return _adapt(regime, aligned, out_index)
```

Rewrite `combine_latest_iter` to the same normalization and to yield `(value_row, index)` (index None per event when positional):

```python
def combine_latest_iter(*values, index=None, emit="when_all"):
    """Yield (value_row, index) aligned rows one at a time (streaming form)."""
    if emit not in ("when_all", "on_any"):
        raise ValueError('combine_latest: emit must be "when_all" or "on_any"')
    streams = _to_streams(values, index)
    kind, keys, vals, positional = _streams_to_keyed(streams, "combine_latest")
    cls = _b._CombineLatestPuller_f64 if kind == "f64" else _b._CombineLatestPuller_i64
    puller = cls(keys, vals, emit == "when_all")
    while True:
        event = puller.next()
        if event is None:
            return
        ev_key, ev_row = event            # existing puller yields (key, row)
        yield ev_row, (None if positional else ev_key)
```
(Confirm the existing puller's `next()` return shape while implementing; adapt the unpack to match. The point is to re-emit as `(value_row, index)`.)

- [ ] **Step 4: Run to verify it passes** — `poetry run pytest tests/test_streams_combine_latest.py -q`.

- [ ] **Step 5: Commit**

```bash
git add screamer/streams.py tests/test_streams_combine_latest.py
git commit -m "feat(streams): combine_latest -> values-first, polymorphic in/out, index optional"
```

---

### Task 3: Migrate `dropna`, `filter`, `select` (+ `_iter`)

**Files:** Modify `screamer/streams.py`; update `tests/test_streams_shape.py`, `tests/test_streams_select.py`.

**Interfaces (new signatures):**
- `dropna(values, index=None, how="any")` -> `(values, index)` / `Stream` / `Node`.
- `filter(values, predicate, index=None)` -> `(values, index)` / `Stream` (eager only; `Node` raises the existing "not supported" message).
- `select(values, columns, index=None)` -> `(values, index)` / `Stream` / `Node`.
- `_iter` twins consume/yield `(value, index)` events (index None for positional).

These are single-stream shape ops, so there is no positional/indexed *mixing* concern. Apply the same wrapper: `Node` -> graph; else classify regime on the single input; normalize to one `Stream`; run the existing filtering/selection logic on `stream.values` (unchanged); the surviving/selected `index` is `stream.index` masked/passed-through the same way as values (or `None` when positional); `_adapt` the output.

- [ ] **Step 1:** Update the tests in `tests/test_streams_shape.py` and `tests/test_streams_select.py` to the new signatures. Representative new-shape tests:

```python
# dropna: raw positional -> (values, None); Stream -> Stream
def test_dropna_raw_positional():
    from screamer.streams import dropna
    values, index = dropna(np.array([1.0, np.nan, 3.0]))
    assert index is None
    np.testing.assert_array_equal(values, [1.0, 3.0])

def test_dropna_indexed_keeps_surviving_index():
    from screamer.streams import dropna
    values, index = dropna(np.array([1.0, np.nan, 3.0]), index=np.array([10, 20, 30]))
    np.testing.assert_array_equal(values, [1.0, 3.0])
    np.testing.assert_array_equal(index, [10, 30])

def test_select_stream_in_stream_out():
    from screamer.streams import select, Stream
    s = Stream(np.array([[10.0, 11.0], [20.0, 21.0]]), np.array([1, 2]))
    out = select(s, 1)
    assert isinstance(out, Stream)
    np.testing.assert_array_equal(out.values, [11.0, 21.0])
    np.testing.assert_array_equal(out.index, [1, 2])
```
Cover: raw/Stream/Node mirroring for dropna and select; `index is None` positional; index masked with values for dropna; `filter(values, predicate)` values-first; `_iter` event `(value, index)`; the graph `dropna`/`select` still compile in a `Dag` (Node path unchanged) and `filter` on a `Node` still raises.

- [ ] **Step 2:** Run to verify failures against old signatures.

- [ ] **Step 3:** Rewrite `dropna`, `filter`, `select`, `dropna_iter`, `filter_iter`, `select_iter` per the wrapper pattern. Keep the numpy masking/projection logic; thread `index` through the same mask/pass-through as `values`; return via `_adapt`. Preserve the `filter`-on-`Node` rejection and the `select`/`dropna` graph dispatch (their `make_combinator_node` kwargs are unchanged: `how`, `columns`).

- [ ] **Step 4:** Run `poetry run pytest tests/test_streams_shape.py tests/test_streams_select.py tests/test_dag_dropna.py tests/test_dag_select.py -q` (graph identity tests must still pass - the Node path is unchanged).

- [ ] **Step 5:** Commit `feat(streams): dropna/filter/select -> values-first, polymorphic in/out`.

---

### Task 4: Migrate `resample` (+ `_iter`), rename `width` -> `every`

**Files:** Modify `screamer/streams.py`, `screamer/dag.py` (dispatch reads `every`); update `tests/test_streams_resample.py`, `tests/test_dag_resample.py`.

**Interfaces:** `resample(values, index=None, *, every=None, count=None, agg="last", origin=0, label="left")` -> `(values, index)` / `Stream` / `Node`, where the returned `index` is the bar labels (never None - resample always produces labels). `resample_iter(events, *, every=None, count=None, ...)` yields `(value, index)`. In `dag.py`, the combinator dispatch reads `kwargs["every"]` (was `width`).

- [ ] **Step 1:** Update `tests/test_streams_resample.py` and `tests/test_dag_resample.py` to `every=` and the new return shape. Add raw/Stream/Node mirroring and: positional input still resamples by `every` on row-number positions; `Stream` in -> `Stream` out with bar-label index; the graph identity (batch == stream == oracle) still holds for `every=`. Keep every existing agg/mode/ohlc/NaN/negative-key/trailing-flush assertion, translated to the new signature.

- [ ] **Step 2:** Run to verify failures.

- [ ] **Step 3:** Rewrite `resample`/`resample_iter`: rename `width` -> `every` (all references), apply the wrapper (Node -> graph with `{"every": ..., ...}`; else regime -> one Stream -> existing bucketing on `stream.values` with `stream.index` or row-number positions -> `(out_values, out_labels)` -> `_adapt`). Update `dag.py` `build()` to read `kwargs["every"]` for the resample node and pass it as the C++ `width` argument (the binding/engine keep the name `width` internally; only the Python kwarg is renamed).

- [ ] **Step 4:** Run `poetry run pytest tests/test_streams_resample.py tests/test_dag_resample.py tests/test_dag_identity.py -q`.

- [ ] **Step 5:** Commit `feat(streams): resample -> values-first + rename width to every`.

---

### Task 5: Migrate `merge`, `split`, `pace`

**Files:** Modify `screamer/streams.py`; update `tests/test_streams_merge.py`, `tests/test_streams_identity.py`.

**Interfaces (raw-array oriented per the spec):**
- `merge(*values, index=None)` -> `(values, sources, index)` (index None when all positional; positional/indexed mixing raises, as in combine_latest). `merge_iter(*values, index=None)` yields `(value, index, source)`.
- `split(values, sources, index=None, n=None)` -> `list[(values, index)]` (each per-source stream; index None when the input index is None).
- `pace(*values, index=None, speed=1.0, sleep=None)` async -> yields `(value, index, source)` events (index None when positional).

- [ ] **Step 1:** Update `tests/test_streams_merge.py` and `tests/test_streams_identity.py` to the new signatures: `merge` returns `(values, sources, index)`; `split(*merge(...))`-style round-trip reconstructs inputs; `pace` yields `(value, index, source)`; the batch == `_iter` == `pace` identity still holds. Add positional (`index is None`) and indexed cases.

- [ ] **Step 2:** Run to verify failures.

- [ ] **Step 3:** Rewrite `merge`/`merge_iter`/`split`/`pace` to the new signatures over the existing cores (the C++ merge/pace drivers are unchanged; only the Python wrapping/order changes). merge/split stay raw-array (no Stream regime). For positional merge, synthesize row-number keys internally and return `index=None`.

- [ ] **Step 4:** Run `poetry run pytest tests/test_streams_merge.py tests/test_streams_identity.py -q`.

- [ ] **Step 5:** Commit `feat(streams): merge/split/pace -> values-first, index optional`.

---

### Task 6: DAG boundary + full rename sweep

**Files:** Modify `screamer/dag.py` (`_as_stream`, `Dag.__call__`/`stream` return shape, `build()` kwargs); grep-sweep `key(s)` -> `index` in remaining user-facing docstrings.

**Interfaces:** `Dag` feeds may be bare value arrays, `Stream`s, or `(values, index)`; `_as_stream` normalizes all three (positional -> engine row-number keys at push). `Dag(...)` outputs follow `align_outputs` in the new `(values, index)` shape (single output -> `(values, index)`; M>1 aligned -> tuple of `(values, index)` pairs; the aligned index replaces the old aligned keys).

- [ ] **Step 1:** Update `tests/test_dag_identity.py` and `tests/test_dag_*.py` to the new return shape (`values, index = dag(...)`), and add a `Dag` fed with `Stream`s. The batch == stream identity assertions stay.

- [ ] **Step 2:** Run to verify failures.

- [ ] **Step 3:** Update `_as_stream` to accept `Stream` / bare array / `(values, index)` and produce the internal `(int64_keys, values)` the engine needs (positional -> `np.arange`). Update `Dag.__call__`/`stream`/`_align_results` to return `(values, index)` shape. Grep `screamer/` for remaining user-facing `key`/`keys` in docstrings and rename to `index`.

- [ ] **Step 4:** Run the full DAG + streams test set: `poetry run pytest tests/test_dag_identity.py tests/test_dag_dropna.py tests/test_dag_select.py tests/test_dag_resample.py -q`.

- [ ] **Step 5:** Commit `feat(dag): accept Stream feeds; return (values, index); key->index sweep`.

---

### Task 7: Documentation update

**Files:** `docs/multistream.md`, `docs/functions_streams/*.md`, `docs/functions_dag/Dag.md`, notebooks `docs/notebooks/07-10`.

- [ ] **Step 1:** Rewrite `docs/multistream.md` to the new model: "a stream is a sequence of values"; the index is an optional ordering coordinate (not a key, not a dict); introduce `Stream`; show the raw/Stream/Node polymorphism and `index=None` positional default. Remove the "(key, value) events" framing and the "order key" tier language where it implies a required key. Keep it dash-free (smartquotes are off).

- [ ] **Step 2:** Update every `docs/functions_streams/*.md` and `functions_dag/Dag.md` example to the new signatures (values-first, `index=`, `every=`, `(values, index)` returns, `Stream` shown once per page). Each example stays one focused, build-executed result.

- [ ] **Step 3:** Update notebooks 07-10 to the new signatures and `key`->`index` prose. Re-run `poetry run pytest --nbmake docs/notebooks/ -q` (all 10 green).

- [ ] **Step 4:** `make docs` builds clean (baseline warnings only); no `key`/dict framing remains in the streams/DAG pages.

- [ ] **Step 5:** Commit `docs: reframe streams around values + optional index (Stream, key->index)`.

---

### Task 8: Final identity re-verification

- [ ] **Step 1:** `poetry run pytest -q` - full suite green (all signatures migrated).
- [ ] **Step 2:** `poetry run pytest --nbmake docs/notebooks/ -q` - all notebooks green.
- [ ] **Step 3:** Spot-check the three regimes end to end (raw arrays, `Stream`, `Node`) for `combine_latest` and `resample`, confirming raw returns `(values, index)` with `index is None` positional, `Stream` returns `Stream`, `Node` builds a graph, and batch == stream.
- [ ] **Step 4:** Commit any test-only fixups; the branch is ready for whole-branch review.

---

## Self-Review

**Spec coverage:** Stream type + interop (Task 1); polymorphic in/out with mirrored returns and `index=None` positional (Tasks 1-6); every combinator + resample migrated values-first (Tasks 2-5); `every` rename (Task 4); merge/split/pace raw-only + `(value, index[, source])` events (Tasks 2-5); DAG boundary + Stream feeds (Task 6); key->index everywhere incl. docs (Tasks 6-7); identity preserved and re-verified (all tasks + Task 8).

**Type consistency:** `Stream(values, index=None)`, `.values`, `.index`, `_regime`/`_to_streams`/`_adapt`, and the `(values, index)` raw return are used identically across all combinators. `_streams_to_keyed` centralizes the positional/indexed/mixed rule for the alignment ops.

**Placeholder scan:** none. The one place implementers must confirm against live code (the existing `combine_latest_iter` puller's `next()` return shape) is called out explicitly in Task 2 Step 3.

**Risk notes:** engine is untouched, so the C++ cores and their identity guarantees are unchanged; every task re-runs the relevant identity tests; the graph (`Node`) path is preserved unchanged in each combinator so the DAG suite is the safety net.
