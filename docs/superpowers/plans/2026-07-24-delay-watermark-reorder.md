# Event-time watermark fix (live Delay/merge/resample) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make a live `Pipeline` with a `Delay` feeding a `CombineLatest` merge produce output identical to batch, by adding an event-time watermark to the DAG that gates a reorder buffer at the merge.

**Architecture:** Add `on_watermark(Index)` to the `Sink` protocol. A watermark is a monotone lower bound on future frame indices; it flows the same topology as data. Pass-through nodes forward it, `Delay` shifts it by `+duration`, `Filter`/`DropNa` forward it past dropped frames, `Resample` closes windows on it, and `CombineLatest` buffers incoming frames and releases them into its existing as-of logic in global index order once the minimum per-port watermark reaches them. Live thereby reproduces the global-index-order delivery that batch already gets from `MergeSource`.

**Tech Stack:** C++17 header-only DAG nodes under `include/screamer/dag/`, pybind11 bindings, Python `screamer.dag` / `screamer.streams` API, pytest.

## Global Constraints

- **`batch == live` is the north star.** Every behavioral test must assert the live path produces the same `(index, value)` output as the batch path.
- **All operator logic in C++.** Python is thin bindings only. No algorithm in Python/numpy.
- **After any C++ change run `make install-dev`** (not just `make build`), or Python imports a stale binding.
- **Run the full suite** (`poetry run python -m pytest -q`) at the end of every task; ops appear in many parametrized sweeps.
- **Fail loud.** The reorder buffer never silently drops or grows unbounded; it raises on overflow.
- **No version-file edits.** Version bumps are `make patch/minor/major`, user-run only.
- **No em-dashes** in any docs or comments (ASCII hyphens only).
- Commit after each task. Footer:

  ```
  Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_018q4wFbrQaLrzUFc1H5NpJx
  ```
  Do NOT push.

## File Structure

- `include/screamer/dag/frame.h` - `Sink` gains `virtual void on_watermark(Index w)` (default no-op).
- `include/screamer/dag/broadcast.h` - fan-out forwards the watermark to all sinks.
- `include/screamer/dag/delay_node.h` - forwards `on_watermark(w + duration)`.
- `include/screamer/dag/combine_latest_node.h` - per-port watermark + reorder buffer + release rule + overflow cap; the one node with real new state.
- `include/screamer/dag/dropna_node.h` - forward watermark; forward the index of a dropped frame as a watermark.
- `include/screamer/dag/filter_node.h` - per-port watermark forwarding through its internal coalescing gate.
- `include/screamer/dag/select_node.h` - forward watermark unchanged.
- `include/screamer/dag/resample_node.h`, `resample_generic_node.h` - close windows on `on_watermark`, unifying `advance()`.
- The compute functor DAG node header(s) - forward watermark (found by grep in Task 1).
- `include/screamer/dag/compiled_graph.h` - `advance(now)` injects `on_watermark(now)` at inputs; `flush()` injects `+INF`; pass the overflow cap through at wiring.
- `screamer/dag.py` and the dag bindings - surface an optional `max_pending` on `Pipeline`.
- `docs/functions_streams/Delay.md` - remove the limitation; document the corrected behavior.
- Tests: `tests/test_dag_watermark.py` (new), plus additions to `tests/test_streams_delay.py`.

---

### Task 1: Watermark protocol + Delay shift + CombineLatest reorder buffer (the core fix)

**Files:**
- Modify: `include/screamer/dag/frame.h` (add `on_watermark` to `Sink`)
- Modify: `include/screamer/dag/broadcast.h` (forward to all)
- Modify: `include/screamer/dag/delay_node.h` (shift watermark)
- Modify: `include/screamer/dag/combine_latest_node.h` (reorder buffer)
- Modify: the compute functor DAG node header and any other pass-through `Sink` subclass (forward watermark)
- Test: `tests/test_dag_watermark.py` (new)

**Interfaces:**
- Produces: `Sink<Index>::on_watermark(Index w)` - "no future frame on this edge has index < w"; default no-op in the base, overridden to forward by pass-through nodes and to gate by `CombineLatestNode`.
- Consumes: existing `Sink::push`, `Sink::flush`, `screamer::streams::CombineLatest` (`on_event`, `latest`).

- [ ] **Step 1: Write the failing flagship test.** In `tests/test_dag_watermark.py`:

```python
import numpy as np
import pytest
from screamer.dag import Input, Pipeline
from screamer.streams import CombineLatest, Delay


def _fused_delay_merge_batch_vs_live(duration, idx, vals):
    """Build Pipeline([x], [CombineLatest()(x, Delay(d)(x))]); return (batch, live)
    each as a sorted list of (index, tuple(row)). Live pushes each event then flushes."""
    x = Input("x")
    pipe = Pipeline([x], [CombineLatest()(x, Delay(duration)(x))])

    batch_v, batch_i = pipe((vals, idx))          # batch: whole arrays
    sess = pipe.live()
    for t, v in zip(idx.tolist(), vals.tolist()):
        sess.push("x", int(t), float(v))
    sess.flush()
    live_v, live_i = sess.result()

    def as_rows(v, i):
        v = np.asarray(v); i = np.asarray(i)
        order = np.argsort(i, kind="stable")
        return [(int(i[k]), tuple(np.ravel(v[k]).tolist())) for k in order]
    return as_rows(batch_v, batch_i), as_rows(live_v, live_i)


def test_fused_delay_merge_batch_equals_live_regular():
    idx = (np.arange(12, dtype=np.int64)) * 10          # regular grid 0,10,...,110
    vals = np.arange(12, dtype=float)
    batch, live = _fused_delay_merge_batch_vs_live(30, idx, vals)
    assert live == batch
```

- [ ] **Step 2: Run it, expect failure** (the bug: live misaligns the delayed port).

Run: `make install-dev && poetry run python -m pytest tests/test_dag_watermark.py -q`
Expected: FAIL - `live != batch` (the delayed events are aligned against stale values).

- [ ] **Step 3: Add `on_watermark` to the `Sink` base.** In `include/screamer/dag/frame.h`, inside `struct Sink`, after the `flush()` declaration:

```cpp
    // Event-time watermark: no future frame delivered to this sink will have
    // index < w. Monotone non-decreasing. Default is a no-op so terminal sinks
    // consume it; pass-through nodes forward it, gating nodes override.
    virtual void on_watermark(Index /*w*/) {}
```

- [ ] **Step 4: Forward the watermark in `Broadcast`.** In `include/screamer/dag/broadcast.h`, add to the class body:

```cpp
    void on_watermark(Index w) override { for (auto* s : sinks_) s->on_watermark(w); }
```

- [ ] **Step 5: Shift the watermark in `DelayNode`.** In `include/screamer/dag/delay_node.h`, add after `flush()`:

```cpp
    void on_watermark(Index w) override { downstream_.on_watermark(w + duration_); }
```

- [ ] **Step 6: Forward the watermark in every pass-through Sink node.** Grep for the compute functor DAG node and any other `class .* : public Sink<Index>` with a `downstream_`:

Run: `grep -rln "public Sink<Index>" include/screamer/dag/`

For each pass-through node that holds a single `Sink<Index>& downstream_` (the functor node, `SelectNode`; `DropNaNode` is done in Task 2), add:

```cpp
    void on_watermark(Index w) override { downstream_.on_watermark(w); }
```

- [ ] **Step 7: Add the reorder buffer to `CombineLatestNode`.** In `include/screamer/dag/combine_latest_node.h`:

Add includes: `#include <queue>`, `#include <cstdint>`, `#include <limits>`.

Add members (after `buffered_row_`):

```cpp
    // Per-port watermark (INT64_MIN = port not yet seen) and a min-heap of frames
    // waiting until the merge is safe to apply them in global index order.
    std::vector<Index> wm_;
    struct Pending { Index index; std::uint32_t port; double value; };
    struct PendGreater {
        bool operator()(const Pending& a, const Pending& b) const {
            if (a.index != b.index) return a.index > b.index;  // min-heap on index
            return a.port  > b.port;                            // tie-break by port
        }
    };
    std::priority_queue<Pending, std::vector<Pending>, PendGreater> pending_;
```

In the constructor initializer list, add `wm_(n, std::numeric_limits<Index>::min())`.

Rename the existing `on_port(std::size_t i, const Frame<Index>& f)` body to a helper that takes an explicit index and value (this is the unchanged as-of + coalescing logic, now fed in guaranteed global order):

```cpp
    void apply_ordered(std::size_t i, Index ev_index, double value) {
        if (cl_.on_event(static_cast<std::uint32_t>(i), value)) {
            const std::vector<double>& row = cl_.latest();
            if (has_buffered_ && ev_index != buffered_index_) {
                downstream_.push(Frame<Index>{buffered_index_, buffered_row_.data(), n_});
            }
            buffered_index_ = ev_index;
            std::copy(row.begin(), row.end(), buffered_row_.begin());
            has_buffered_ = true;
        }
    }
```

Replace `on_port` with an enqueue-and-release path, and add a watermark path and the release routine:

```cpp
    void on_port(std::size_t i, const Frame<Index>& f) {
        assert(f.width == 1);
        if (wm_[i] < f.index) wm_[i] = f.index;         // a data frame advances the port watermark
        pending_.push(Pending{f.index, static_cast<std::uint32_t>(i), f.values[0]});
        release();
    }

    void on_port_watermark(std::size_t i, Index w) {
        if (wm_[i] < w) wm_[i] = w;
        release();
    }

    // Apply every pending frame the merge can now prove is safe (index <= the
    // minimum per-port watermark), in global index order, then forward the
    // settled watermark downstream.
    void release() {
        Index low = wm_[0];
        for (std::size_t j = 1; j < n_; ++j) low = std::min(low, wm_[j]);
        while (!pending_.empty() && pending_.top().index <= low) {
            Pending p = pending_.top(); pending_.pop();
            apply_ordered(p.port, p.index, p.value);
        }
        if (low != std::numeric_limits<Index>::min()) downstream_.on_watermark(low);
    }
```

Add the `Port::on_watermark` override (in the inner `Port` struct):

```cpp
        void on_watermark(Index w) override { node.on_port_watermark(idx, w); }
```

In `flush_downstream(i)`, set the flushing port's watermark to the max before the settle, so the final drain releases everything in order. Change the body to:

```cpp
    void flush_downstream(std::size_t i) {
        wm_[i] = std::numeric_limits<Index>::max();   // this port will send no more
        if (!flushed_[i]) { flushed_[i] = true; ++flushed_count_; }
        release();                                    // drain what is now safe
        if (flushed_count_ < n_) return;

        if (has_buffered_) {
            downstream_.push(Frame<Index>{buffered_index_, buffered_row_.data(), n_});
            has_buffered_ = false;
        }
        downstream_.flush();

        std::fill(flushed_.begin(), flushed_.end(), false);
        flushed_count_ = 0;
    }
```

In `reset()`, add:

```cpp
        std::fill(wm_.begin(), wm_.end(), std::numeric_limits<Index>::min());
        while (!pending_.empty()) pending_.pop();
```

- [ ] **Step 8: Confirm flush drains the buffer (no compiled_graph change needed).** The flush cascade already reaches every merge port: `compiled_graph::flush()` calls `input_sinks_[i]->flush()`, which propagates through `Broadcast`/`Delay` to each `CombineLatestNode::Port::flush()` -> `flush_downstream`. Step 7's `flush_downstream` sets that port's watermark to `MAX` and calls `release()`, so once all ports have flushed the reorder buffer drains in global index order before the final coalesced row is emitted. Verify this path by inspection; do not add a separate watermark injection to `flush()` (it would be redundant with `flush_downstream`).

Note on batch: in `run_batch`, `MergeSource` feeds the merge in global index order, and each data frame advances its own port's watermark, so the reorder buffer releases incrementally and stays bounded by the inter-event skew. A merge input that is entirely empty in batch is the one case where frames buffer until flush (bounded by the batch size, the same memory order batch already uses); Task 4's cap covers the degenerate stalled case.

- [ ] **Step 9: Build and run the flagship test.**

Run: `make install-dev && poetry run python -m pytest tests/test_dag_watermark.py -q`
Expected: PASS.

- [ ] **Step 10: Add irregular-feed and multi-duration cases** to `tests/test_dag_watermark.py`:

```python
def test_fused_delay_merge_batch_equals_live_irregular():
    rng = np.random.default_rng(1)
    idx = np.cumsum(rng.integers(1, 9, size=150)).astype(np.int64)
    vals = rng.standard_normal(150)
    for d in (1, 4, 25, 100):
        batch, live = _fused_delay_merge_batch_vs_live(d, idx, vals)
        assert live == batch, f"mismatch at duration {d}"


def test_watermark_propagates_through_a_functor():
    # A functor between the input and a delayed merge must forward the watermark,
    # or the merge would stall and diverge from batch.
    from screamer import RollingMean
    x = Input("x")
    pipe = Pipeline([x], [CombineLatest()(RollingMean(3)(x), Delay(20)(x))])
    idx = (np.arange(15, dtype=np.int64)) * 10
    vals = np.arange(15, dtype=float)
    bv, bi = pipe((vals, idx))
    s = pipe.live()
    for t, v in zip(idx.tolist(), vals.tolist()):
        s.push("x", int(t), float(v))
    s.flush()
    lv, li = s.result()
    ob, ol = np.argsort(bi, kind="stable"), np.argsort(li, kind="stable")
    np.testing.assert_array_equal(np.asarray(li)[ol], np.asarray(bi)[ob])
    np.testing.assert_allclose(np.asarray(lv)[ol], np.asarray(bv)[ob])
```

- [ ] **Step 11: Run the full suite** (nothing else should regress; the merge now reorders but for in-order input it is a passthrough).

Run: `make install-dev && poetry run python -m pytest -q`
Expected: PASS (same count as before, plus the new tests).

- [ ] **Step 12: Commit.**

```bash
git add include/screamer/dag/ tests/test_dag_watermark.py
git commit -m "feat(dag): watermark protocol + CombineLatest reorder buffer for correct live Delay->merge"
```

---

### Task 2: Filter and DropNa forward the watermark past dropped frames

**Files:**
- Modify: `include/screamer/dag/dropna_node.h`
- Modify: `include/screamer/dag/filter_node.h`
- Test: add to `tests/test_dag_watermark.py`

**Interfaces:**
- Consumes: `Sink::on_watermark` from Task 1.
- Produces: nothing new; `DropNaNode` and `FilterNode` now forward watermarks (and the index of a dropped frame) downstream.

- [ ] **Step 1: Write the failing test.** A `DropNa` between the input and a delayed merge port drops NaN rows; if the dropped frame's index does not advance the merge watermark, the merge stalls and diverges from batch. In `tests/test_dag_watermark.py`:

```python
def test_dropna_before_delayed_merge_does_not_stall():
    from screamer.dag import Input, Pipeline
    from screamer.streams import CombineLatest, Delay, Dropna
    x = Input("x")
    # x is cleaned by Dropna, delayed, and merged against raw x.
    pipe = Pipeline([x], [CombineLatest()(x, Delay(20)(Dropna()(x)))])
    idx = (np.arange(20, dtype=np.int64)) * 10
    vals = np.arange(20, dtype=float)
    vals[5] = np.nan      # a dropped row: its index must still advance the watermark
    vals[11] = np.nan
    bv, bi = pipe((vals, idx))
    s = pipe.live()
    for t, v in zip(idx.tolist(), vals.tolist()):
        s.push("x", int(t), float(v))
    s.flush()
    lv, li = s.result()
    ob, ol = np.argsort(bi, kind="stable"), np.argsort(li, kind="stable")
    np.testing.assert_array_equal(np.asarray(li)[ol], np.asarray(bi)[ob])
    np.testing.assert_allclose(np.asarray(lv)[ol], np.asarray(bv)[ob], equal_nan=True)
```

(If the `Dropna` stream operator name differs, confirm it from `tests/test_streams_dropna.py` and use the exact name.)

- [ ] **Step 2: Run it, expect failure or stall-divergence.**

Run: `make install-dev && poetry run python -m pytest tests/test_dag_watermark.py::test_dropna_before_delayed_merge_does_not_stall -q`
Expected: FAIL (`live != batch`; the merge held the delayed events because the dropped indices never advanced its watermark).

- [ ] **Step 3: Forward-on-drop in `DropNaNode`.** In `include/screamer/dag/dropna_node.h`, change `push` so a dropped frame still advances downstream time, and add watermark forwarding:

```cpp
    void push(const Frame<Index>& f) override {
        bool any_nan = false;
        bool all_nan = f.width > 0;
        for (std::size_t i = 0; i < f.width; ++i) {
            if (screamer::isnan2(f.values[i])) any_nan = true;
            else                               all_nan = false;
        }
        bool drop = how_all_ ? all_nan : any_nan;
        if (!drop) downstream_.push(f);
        else       downstream_.on_watermark(f.index);   // dropped: advance time only
    }

    void on_watermark(Index w) override { downstream_.on_watermark(w); }
```

- [ ] **Step 4: Forward the watermark through `FilterNode`.** `FilterNode` is a fan-in (two ports, `CombineLatest(2, when_all=true)` coalescing). Add a port watermark path that advances a per-port watermark and, when the mask gate drops a settled row, forwards its index downstream as a watermark. Add a member `std::vector<Index> wm_;` (declared after the existing members) and initialize it in the constructor initializer list as `wm_(2, std::numeric_limits<Index>::min())`. Add `#include <limits>`. Add to the inner `Port` struct:

```cpp
        void on_watermark(Index w) override { node.on_port_watermark(idx, w); }
```

Add to the class:

```cpp
    void on_port_watermark(std::size_t i, Index w) {
        if (wm_[i] < w) wm_[i] = w;
        Index low = std::min(wm_[0], wm_[1]);
        if (low != std::numeric_limits<Index>::min()) downstream_.on_watermark(low);
    }
```

In `emit_buffered()`, when the row is dropped by the gate, forward its index as a watermark so downstream time still advances:

```cpp
    void emit_buffered() {
        if (has_buffered_) {
            if (buffered_mask_ != 0.0 && !screamer::isnan2(buffered_mask_))
                downstream_.push(Frame<Index>{buffered_index_, &buffered_data_, 1});
            else
                downstream_.on_watermark(buffered_index_);   // dropped: advance time only
        }
        has_buffered_ = false;
    }
```

In `reset()`, add `std::fill(wm_.begin(), wm_.end(), std::numeric_limits<Index>::min());`.

- [ ] **Step 5: Build and run the test.**

Run: `make install-dev && poetry run python -m pytest tests/test_dag_watermark.py -q`
Expected: PASS.

- [ ] **Step 6: Full suite.**

Run: `poetry run python -m pytest -q`
Expected: PASS.

- [ ] **Step 7: Commit.**

```bash
git add include/screamer/dag/dropna_node.h include/screamer/dag/filter_node.h tests/test_dag_watermark.py
git commit -m "feat(dag): Filter/DropNa forward watermark past dropped frames"
```

---

### Task 3: Unify `advance(now)` onto the watermark, close Resample windows on it

**Files:**
- Modify: `include/screamer/dag/resample_node.h` (and `resample_generic_node.h` if it has its own `advance`)
- Modify: `include/screamer/dag/compiled_graph.h` (`advance(now)` injects a watermark)
- Test: add to `tests/test_dag_watermark.py`

**Interfaces:**
- Consumes: `Sink::on_watermark`; the existing `ResampleNode::advance(Index)` logic.
- Produces: `advance(now)` now flows through the whole graph as a watermark; `ResampleNode::on_watermark(w)` closes windows.

- [ ] **Step 1: Write the failing test** for idle-port release via `advance(now)`. A delayed merge whose other input goes idle should release buffered rows when the user advances logical time:

```python
def test_advance_releases_idle_delayed_buffer():
    from screamer.dag import Input, Pipeline
    from screamer.streams import CombineLatest, Delay
    a, b = Input("a"), Input("b")
    # b is delayed and merged against a. a stops early; advance(now) must let the
    # buffered delayed-b rows settle against a's last value.
    pipe = Pipeline([a, b], [CombineLatest()(a, Delay(5)(b))])
    s = pipe.live()
    s.push("a", 0, 10.0)
    s.push("b", 0, 1.0)     # delayed to index 5
    s.push("b", 3, 2.0)     # delayed to index 8
    s.advance(100)          # a is idle; advancing time releases the buffered rows
    v, i = s.result()
    # both delayed b rows are now settled against a-as-of (a=10.0 at every index >=0)
    assert i is not None and len(np.asarray(i)) >= 2
```

Also add a regression assertion that the existing resample `advance` still works: pick one representative from `tests/test_streams_resample.py` that calls `.advance(...)` and confirm it still passes after the change (it must).

- [ ] **Step 2: Run it, expect failure** (today `advance(now)` only reaches resample nodes, so the merge never releases).

Run: `make install-dev && poetry run python -m pytest tests/test_dag_watermark.py::test_advance_releases_idle_delayed_buffer -q`
Expected: FAIL (no rows released; `result()` is empty or short).

- [ ] **Step 3: Close Resample windows on the watermark.** In `include/screamer/dag/resample_node.h`, add (the body reuses the existing `advance` logic):

```cpp
    void on_watermark(Index w) override {
        advance(w);
        downstream_.on_watermark(w);
    }
```

Do the same in `resample_generic_node.h` if it defines its own `advance`.

- [ ] **Step 4: Make `advance(now)` inject a watermark at the inputs.** In `include/screamer/dag/compiled_graph.h`, change `advance` from the direct per-resample calls to a graph-wide watermark injection:

```cpp
    void advance(std::int64_t now) {
        for (auto* s : input_sinks_) if (s) s->on_watermark(now);
    }
```

The `advance_resamples_` / `advance_generic_resamples_` lists and their population can be removed (the watermark reaches resample nodes through the graph now). If removing them is noisy, leave the vectors unused; the plan's reviewer will flag dead members.

- [ ] **Step 5: Build and run the new test and the resample regression.**

Run: `make install-dev && poetry run python -m pytest tests/test_dag_watermark.py tests/test_streams_resample.py -q`
Expected: PASS (idle-release works; resample advance behavior unchanged).

- [ ] **Step 6: Full suite.**

Run: `poetry run python -m pytest -q`
Expected: PASS.

- [ ] **Step 7: Commit.**

```bash
git add include/screamer/dag/resample_node.h include/screamer/dag/resample_generic_node.h include/screamer/dag/compiled_graph.h tests/test_dag_watermark.py
git commit -m "feat(dag): unify advance(now) onto the watermark; Resample closes on it"
```

---

### Task 4: Overflow cap on the reorder buffer

**Files:**
- Modify: `include/screamer/dag/combine_latest_node.h` (cap + throw)
- Modify: `include/screamer/dag/compiled_graph.h`, `include/screamer/dag/graph.h`, `screamer/dag.py` and the dag binding (thread an optional `max_pending` through to `CombineLatestNode`)
- Test: add to `tests/test_dag_watermark.py`

**Interfaces:**
- Consumes: the reorder buffer from Task 1.
- Produces: `CombineLatestNode` constructor gains a `std::size_t max_pending` parameter (default a large constant); exceeding it raises `std::runtime_error` (Python `RuntimeError`).

- [ ] **Step 1: Write the failing test.** A never-advancing sibling makes the buffer grow without bound; a small cap must raise:

```python
def test_reorder_buffer_overflow_raises():
    from screamer.dag import Input, Pipeline
    from screamer.streams import CombineLatest, Delay
    a, b = Input("a"), Input("b")
    pipe = Pipeline([a, b], [CombineLatest()(a, Delay(1)(b))], max_pending=8)
    s = pipe.live()
    with pytest.raises(RuntimeError):
        for t in range(100):
            s.push("b", t, float(t))    # a never advances -> b's delayed rows pile up
```

- [ ] **Step 2: Run it, expect failure** (no cap yet; either it accepts the kwarg with no effect or `Pipeline` rejects `max_pending`).

Run: `make install-dev && poetry run python -m pytest tests/test_dag_watermark.py::test_reorder_buffer_overflow_raises -q`
Expected: FAIL.

- [ ] **Step 3: Add the cap to `CombineLatestNode`.** Give the constructor a `std::size_t max_pending` parameter stored as `max_pending_`, and in `on_port`, after `pending_.push(...)`:

```cpp
        if (pending_.size() > max_pending_)
            throw std::runtime_error(
                "CombineLatest reorder buffer overflow: an input has not advanced; "
                "a stream is stalled. Call advance(now) or check the feed.");
```

- [ ] **Step 4: Thread `max_pending` through the graph.** Add a `std::size_t max_pending` field to the `CombineLatest` `NodeSpec` (default `1000000`) in `graph.h`'s `add_combine_latest`, pass it into the `CombineLatestNode` constructor in `compiled_graph.h`'s `CombineLatest` wiring case, and surface an optional `max_pending=1_000_000` kwarg on `Pipeline` in `screamer/dag.py` (plumbed to `gb.compile(...)` / the graph builder). Read the current `add_combine_latest` signature and `Pipeline.__init__` to place the parameter; keep the default so all existing call sites are unaffected.

- [ ] **Step 5: Build and run the test.**

Run: `make install-dev && poetry run python -m pytest tests/test_dag_watermark.py::test_reorder_buffer_overflow_raises -q`
Expected: PASS.

- [ ] **Step 6: Full suite** (the default cap must not trip any existing pipeline).

Run: `poetry run python -m pytest -q`
Expected: PASS.

- [ ] **Step 7: Commit.**

```bash
git add include/screamer/dag/ screamer/dag.py bindings/ tests/test_dag_watermark.py
git commit -m "feat(dag): bounded reorder buffer, raises on overflow (default 1e6)"
```

---

### Task 5: Documentation

**Files:**
- Modify: `docs/functions_streams/Delay.md` (remove the limitation, document the corrected behavior)
- Modify: `docs/multistream.md` if it references the limitation

- [ ] **Step 1: Rewrite the Delay limitation.** In `docs/functions_streams/Delay.md`, replace the `## Limitations` section (the "planned follow-on" reorder-buffer note) with a short statement that a `Delay` feeding a live `CombineLatest` is now correct: the merge holds future-dated events in a bounded reorder buffer and releases them in index order once the other inputs advance, so batch and live agree. Note `advance(now)` releases buffered rows when a sibling stream is idle, and that a stalled stream trips the reorder cap with a clear error. No em-dashes.

- [ ] **Step 2: Grep for other mentions.** `grep -rn "reorder\|live-merge\|planned follow-on" docs/` and update any stale reference (for example the `DelayNode` header comment in `include/screamer/dag/delay_node.h` that says "See Delay.md for the live-merge-fusion limitation").

- [ ] **Step 3: Commit.**

```bash
git add docs/ include/screamer/dag/delay_node.h
git commit -m "docs(delay): the live Delay->merge limitation is fixed"
```

---

### Task 6: Full verification and batch==live sweep

**Files:**
- Test: add a parametrized sweep to `tests/test_dag_watermark.py`

- [ ] **Step 1: Add a composition sweep** asserting batch==live across delay durations, feed regularities, and a `when_all` merge:

```python
import itertools

@pytest.mark.parametrize("d,emit", list(itertools.product([1, 7, 50], ["on_any", "when_all"])))
def test_fused_delay_merge_sweep(d, emit):
    from screamer.dag import Input, Pipeline
    from screamer.streams import CombineLatest, Delay
    rng = np.random.default_rng(hash((d, emit)) % (2**32))
    idx = np.cumsum(rng.integers(1, 6, size=120)).astype(np.int64)
    vals = rng.standard_normal(120)
    x = Input("x")
    cl = CombineLatest(emit=emit) if emit == "when_all" else CombineLatest()
    pipe = Pipeline([x], [cl(x, Delay(d)(x))])
    bv, bi = pipe((vals, idx))
    s = pipe.live()
    for t, v in zip(idx.tolist(), vals.tolist()):
        s.push("x", int(t), float(v))
    s.flush()
    lv, li = s.result()
    ob, ol = np.argsort(bi, kind="stable"), np.argsort(li, kind="stable")
    np.testing.assert_array_equal(np.asarray(li)[ol], np.asarray(bi)[ob])
    np.testing.assert_allclose(np.asarray(lv)[ol], np.asarray(bv)[ob], equal_nan=True)
```

(Confirm the `emit=` value for a when-all merge from `tests/test_streams_combine_latest.py`; use the exact spelling.)

- [ ] **Step 2: Run the whole suite and build the docs.**

Run: `make install-dev && poetry run python -m pytest -q && make docs`
Expected: all tests pass; docs build clean (only the 2 known pre-existing sphinx_exec_code warnings).

- [ ] **Step 3: Commit.**

```bash
git add tests/test_dag_watermark.py
git commit -m "test(dag): batch==live sweep across durations, feeds, and emit modes"
```
