# Delay + forecast_pairs Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `Delay(duration)` (a C++ dag stream op that re-stamps each event's index by a time offset) and `forecast_pairs(X, y, count=|duration=)` (a Python supervised-pairing helper that builds forecasting training sets by lagging features to align with a future target).

**Architecture:** `Delay` is a stateless dag node (`DelayNode : Sink<Index>`) that forwards each frame with `index + duration`, wired into the graph builder exactly like `Resample`. `forecast_pairs` is thin Python in a new `screamer.supervised` module that composes existing C++ nodes (`Lag` for `count=`, `Delay` for `duration=`, plus `CombineLatest`) and adds no operator compute.

**Tech Stack:** C++17 dag/streams engine (`include/screamer/dag/`), pybind11 (`bindings/bindings_dag.cpp`), Python surface (`screamer/streams.py`, `screamer/dag.py`, new `screamer/supervised.py`), pytest.

## Global Constraints

- `Lag(window_size)` is unchanged; `Delay` and `forecast_pairs` are additive. No breaking change.
- `Delay(duration)`: `duration` positional, numeric, **in index units** (like `Resample`'s numeric window; no string/calendar parsing). Re-stamp `(t, v) -> (t + duration, v)`. Lossless, 1:1, order-preserving, no NaN warmup.
- `Delay` **requires an explicit index**: calling it on a bare array or with `index=None` is a `TypeError` (duration is time-based; there is nothing to shift against). `Lag` never needs an index.
- The engine is **int64-indexed**; a fractional index is rejected loudly (reuse `_int64_index`).
- `forecast_pairs(X, y, count=N|duration=D)`: exactly one of `count`/`duration` (a one-of; both or neither is an error). `count=` always works; `duration=` inherits `Delay`'s index requirement. Pairs features from the past with the target now ("features predict the target `horizon` ahead"). Caller contract: **y must be causal** (known as-of its own index). Output preserves alignment with leading warmup NaN on the shifted features (screamer idiom); `dropna=True` returns NaN-free arrays; also returns an `as_of` index (each example's completion time).
- All operator compute stays in C++; `forecast_pairs` is orchestration only (`Lag`/`Delay`/`CombineLatest` are C++ nodes).
- `Delay` lives in `screamer.supervised`? No: `Delay` is a core stream op (top-level `screamer.Delay`, `topics: [streams]`); `forecast_pairs` lives in `screamer.supervised`.
- After C++ changes: `make install-dev`. Regenerate docs registry: `poetry run python devtools/build_help_registry.py` then `poetry run python devtools/build_topic_pages.py`. Regenerate the package init: `make regen-init`. Final docs gate: `make docs` (exit 0).
- No em-dashes in prose/comments/docstrings (ASCII hyphens). Commit as `simu.ai <claude@sitmo.com>` with the footer:

      Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
      Claude-Session: https://claude.ai/code/session_018q4wFbrQaLrzUFc1H5NpJx

  Do not edit version files. Do not push.

## Known limitation to document (not a bug to fix here)

A `Delay` feeding a downstream merge (`CombineLatest`) fused inside a **single live compiled Pipeline** emits the delayed event eagerly, so the merge would pair it against a not-yet-advanced other input (stale as-of). This plan's `Delay` is fully correct for the direct-call / single-chain use, which is what `forecast_pairs` uses (it composes `Delay` and `CombineLatest` as separate batch calls, where batch `CombineLatest` sees all events at once). The fused-live-merge case needs a reorder buffer and is an explicit follow-on; document it in `Delay.md` and do not wire around it here.

---

## File Structure

- `include/screamer/dag/delay_node.h` (create) - `DelayNode<Index>`, the stateless re-stamp node.
- `include/screamer/dag/graph.h` (modify) - `NodeKind::Delay`, `NodeSpec.delay_duration`, `GraphBuilder::add_delay`.
- `include/screamer/dag/compiled_graph.h` (modify) - include the node, width passthrough, wiring case.
- `bindings/bindings_dag.cpp` (modify) - `PyGraphBuilder::add_delay` + its `.def`.
- `screamer/streams.py` (modify) - `delay()` function + `Delay` class + `_delay_via_cpp` + `__all__`.
- `screamer/dag.py` (modify) - `_compile_cpp` gains an `elif name == "Delay"` branch.
- `screamer/__init__.py` (modify/regenerated) - export `Delay`.
- `screamer/supervised.py` (create) - `forecast_pairs`.
- `docs/functions_streams/Delay.md`, `docs/functions_supervised/forecast_pairs.md` (create) - docs pages.
- `tests/test_streams_delay.py`, `tests/test_dag_delay.py`, `tests/test_supervised_forecast_pairs.py` (create).
- `CHANGELOG.md` (modify).

---

## Task 1: `Delay` C++ node, wiring, binding, and Python surface

**Files:**
- Create: `include/screamer/dag/delay_node.h`
- Modify: `include/screamer/dag/graph.h`, `include/screamer/dag/compiled_graph.h`, `bindings/bindings_dag.cpp`, `screamer/streams.py`, `screamer/dag.py`
- Test: `tests/test_streams_delay.py`, `tests/test_dag_delay.py`

**Interfaces:**
- Produces: `screamer.Delay(duration)` callable; `Delay(duration)(values, index)` returns `(values, index + duration)`. Usable as a `Node` in a `Pipeline`. Requires an explicit index.

- [ ] **Step 1: Write the failing eager test** (`tests/test_streams_delay.py`)

```python
import numpy as np
import pytest
from screamer import Delay


def test_delay_shifts_index_leaves_values():
    vals = np.array([1.0, 2.0, 3.0])
    idx = np.array([0, 7, 14], dtype=np.int64)
    v, i = Delay(5)(vals, idx)
    np.testing.assert_array_equal(v, vals)           # values unchanged
    np.testing.assert_array_equal(i, [5, 12, 19])    # index + duration


def test_delay_requires_explicit_index():
    with pytest.raises(TypeError):
        Delay(5)(np.array([1.0, 2.0, 3.0]))          # no index -> error


def test_delay_on_regular_grid_matches_index_shift():
    vals = np.arange(10.0)
    idx = np.arange(10, dtype=np.int64) * 100         # 100-unit grid
    v, i = Delay(300)(vals, idx)                       # 3-step delay
    np.testing.assert_array_equal(i, idx + 300)
    np.testing.assert_array_equal(v, vals)
```

- [ ] **Step 2: Run to verify failure**

Run: `poetry run python -m pytest tests/test_streams_delay.py -q`
Expected: FAIL (`cannot import name 'Delay'`).

- [ ] **Step 3: Create `include/screamer/dag/delay_node.h`**

```cpp
#ifndef SCREAMER_DAG_DELAY_NODE_H
#define SCREAMER_DAG_DELAY_NODE_H

#include <cstdint>
#include "screamer/dag/frame.h"

namespace screamer { namespace dag {

// Stateless index re-stamp. Forwards each frame with index shifted by `duration`,
// values untouched. Lossless, 1:1, order-preserving (a constant positive shift keeps
// events monotonic), no warmup. See Delay.md for the live-merge-fusion limitation.
template <class Index>
class DelayNode : public Sink<Index> {
public:
    DelayNode(std::int64_t duration, Sink<Index>& downstream)
        : duration_(duration), downstream_(downstream) {}

    void push(const Frame<Index>& f) override {
        Frame<Index> out{ static_cast<Index>(f.index + duration_), f.values, f.width };
        downstream_.push(out);
    }

    void flush() override { downstream_.flush(); }
    void reset() override {}                      // stateless

    std::size_t n_in()  const override { return 1; }
    std::size_t n_out() const override { return 1; }

private:
    std::int64_t duration_;
    Sink<Index>& downstream_;
};

}} // namespace screamer::dag

#endif // SCREAMER_DAG_DELAY_NODE_H
```

- [ ] **Step 4: Wire into `include/screamer/dag/graph.h`**

Add `Delay` to the `NodeKind` enum:
```cpp
enum class NodeKind { Input, Functor, CombineLatest, DropNa, Select, Resample, Filter, Delay };
```
Add a field to `NodeSpec` (after the `resample` field):
```cpp
std::int64_t delay_duration = 0;
```
Add the builder method next to `add_resample`:
```cpp
std::size_t add_delay(std::vector<std::size_t> inputs, std::int64_t duration) {
    NodeSpec ns{NodeKind::Delay, nullptr, true, false, {}, {}, std::move(inputs)};
    ns.delay_duration = duration;
    spec_.nodes.push_back(std::move(ns));
    return spec_.nodes.size() - 1;
}
```

- [ ] **Step 5: Wire into `include/screamer/dag/compiled_graph.h`**

Add the include near the other dag node includes:
```cpp
#include "screamer/dag/delay_node.h"
```
In the `node_width` switch (the passthrough-width block), add:
```cpp
case NodeKind::Delay: node_width[id] = node_width[nd.inputs[0]]; break;
```
In the node-wiring switch (next to `case NodeKind::Resample:`), add:
```cpp
case NodeKind::Delay: {
    auto dn = std::make_shared<DelayNode<std::int64_t>>(ns.delay_duration, *downstream);
    node_input_sink[id] = [ptr = dn.get()](std::size_t) -> Sink<std::int64_t>* { return ptr; };
    owned_.push_back(dn);
    break;                                        // stateless: not added to reset_nodes_
}
```

- [ ] **Step 6: Bind in `bindings/bindings_dag.cpp`**

Add a method to the `PyGraphBuilder` struct (next to `add_resample`):
```cpp
std::size_t add_delay(std::vector<std::size_t> inputs, std::int64_t duration) {
    return builder.add_delay(std::move(inputs), duration);
}
```
Add its `.def` in the `py::class_<PyGraphBuilder>` block (next to the `add_resample` def):
```cpp
.def("add_delay", [](PyGraphBuilder& b, std::vector<std::size_t> inputs,
                     std::int64_t duration) {
    return b.add_delay(std::move(inputs), duration);
}, py::arg("inputs"), py::arg("duration"))
```

- [ ] **Step 7: Add the Python surface in `screamer/streams.py`**

Add the `delay()` function and `Delay` class (mirroring `resample`/`Resample`; place near them):
```python
def delay(values, index=None, *, duration):
    """Re-stamp each event's index by `duration` (index units); values unchanged.

    Requires an explicit index (duration is time-based). Returns (values, index +
    duration) as a concrete stream, or a graph Node when `values` is a Node."""
    duration = int(duration)
    if is_node(values):
        return make_operator_node(Delay, (values,), {"duration": duration})
    if index is None:
        raise TypeError(
            "Delay requires an explicit index (duration is time-based); "
            "call Delay(duration)(values, index).")
    return _delay_via_cpp((values, index), duration=duration)


def _delay_via_cpp(feed, *, duration):
    from .dag import Input, Pipeline
    src = Input("x")
    node = delay(src, duration=duration)
    dag = Pipeline([src], [node])
    return dag(feed)


class Delay:
    def __init__(self, duration):
        self._duration = int(duration)

    def __call__(self, values, index=None):
        return delay(values, index, duration=self._duration)
```
Add `"Delay"` and `"delay"` to `streams.py`'s `__all__`.

- [ ] **Step 8: Dispatch in `screamer/dag.py` `_compile_cpp`**

In the `elif isinstance(op, tuple) and op[0] == "operator":` block, add a branch next to the `Resample` one:
```python
elif name == "Delay":
    nid = gb.add_delay(inp, int(kwargs["duration"]))
```

- [ ] **Step 9: Add the graph-mode test** (`tests/test_dag_delay.py`)

```python
import numpy as np
from screamer import Delay
from screamer.dag import Input, Pipeline


def test_delay_as_pipeline_node():
    x = Input("x")
    p = Pipeline([x], [Delay(5)(x)])
    v, i = p((np.array([1.0, 2.0, 3.0]), np.array([0, 7, 14], dtype=np.int64)))
    np.testing.assert_array_equal(v, [1.0, 2.0, 3.0])
    np.testing.assert_array_equal(i, [5, 12, 19])
```

- [ ] **Step 10: Build, export, run**

Run: `make install-dev && make regen-init && poetry run python -m pytest tests/test_streams_delay.py tests/test_dag_delay.py -q`
Expected: PASS (all 4 tests). `make regen-init` adds `Delay` to `screamer/__init__.py`; if the generator does not pick it up, add `Delay` (and `delay`) to the streams re-export list it reads.

- [ ] **Step 11: Commit**

```bash
git add include/screamer/dag/delay_node.h include/screamer/dag/graph.h include/screamer/dag/compiled_graph.h bindings/bindings_dag.cpp screamer/streams.py screamer/dag.py screamer/__init__.py tests/test_streams_delay.py tests/test_dag_delay.py
git commit -m "feat(streams): Delay(duration) index re-stamp stream op"
```

---

## Task 2: `Delay` batch==stream check, docs, and help registration

**Files:**
- Modify: `tests/test_streams_delay.py`
- Create: `docs/functions_streams/Delay.md`
- Modify: `CHANGELOG.md`, `screamer/data/help.json` (regenerated)

**Interfaces:**
- Consumes: `Delay` from Task 1.

- [ ] **Step 1: Add the batch==stream + irregular-feed tests** (append to `tests/test_streams_delay.py`)

The live drive uses the session API `push(input, index, value)`, `flush()`, `result()`:
```python
def test_delay_irregular_feed_worked_example():
    # 7s-spaced feed, 5s delay: value at t=7 becomes current at t=12
    vals = np.array([10.0, 11.0, 12.0])
    idx = np.array([0, 7, 14], dtype=np.int64)
    v, i = Delay(5)(vals, idx)
    np.testing.assert_array_equal(list(zip(i.tolist(), v.tolist())),
                                  [(5, 10.0), (12, 11.0), (19, 12.0)])


def test_delay_batch_equals_live():
    from screamer.dag import Input, Pipeline
    rng = np.random.default_rng(0)
    n = 200
    vals = rng.standard_normal(n)
    idx = np.cumsum(rng.integers(1, 9, size=n)).astype(np.int64)   # irregular, increasing
    batch_v, batch_i = Delay(4)((vals, idx))

    x = Input("x")
    pipe = Pipeline([x], [Delay(4)(x)])
    sess = pipe.live()
    for t, val in zip(idx.tolist(), vals.tolist()):
        sess.push("x", int(t), float(val))
    sess.flush()
    live_v, live_i = sess.result()
    # compare as (index, value) sets, sorted by index (drain order is not guaranteed)
    order_b = np.argsort(batch_i, kind="stable")
    order_l = np.argsort(live_i, kind="stable")
    np.testing.assert_array_equal(live_i[order_l], batch_i[order_b])
    np.testing.assert_allclose(live_v[order_l], batch_v[order_b])
```

- [ ] **Step 2: Run to verify** the batch==stream test passes

Run: `poetry run python -m pytest tests/test_streams_delay.py -q -k "irregular or batch_equals_live"`
Expected: PASS. (If the live drain orders differently, sort both by index before comparing; the invariant is the same `(index, value)` set.)

- [ ] **Step 3: Create `docs/functions_streams/Delay.md`**

Front-matter plus body, matching `Resample.md`'s structure:
```markdown
---
name: Delay
title: Delay
kind: class
short: Re-stamp each event's index by a fixed time offset (a latency line).
topics:
- streams
covers:
- delay
---

# `Delay`

`Delay(duration)` shifts every event's index by `duration` (in index units) and
leaves its value unchanged: event `(t, v)` becomes `(t + duration, v)`. It is the
time-based counterpart of `Lag`, which shifts by a fixed number of events. On a
regular grid the two coincide (`duration = N * step` equals `Lag(N)`); on an
irregular feed `Delay` shifts by wall-time and `Lag` shifts by event count.

The shift is lossless, one output per input, order-preserving, and starts `duration`
late (no NaN warmup). `duration` is numeric in index units (a millisecond index uses
`Delay(600_000)` for a 10-minute delay); there is no calendar parsing.

`Delay` requires an explicit index (there is nothing to shift against without one);
calling it on a bare array is a `TypeError`.

## Parameters

- `duration`: the index offset to add to every event, numeric, in index units.

## Limitations

A `Delay` feeding a downstream merge (`CombineLatest`) inside a single live
`Pipeline` emits the delayed event eagerly, so the merge would align it against a
not-yet-advanced input. For an as-of alignment across a delay, apply `Delay` and
`CombineLatest` as separate calls (batch `CombineLatest` sees all events), which is
what `forecast_pairs` does. A fused-live-merge form needs a reorder buffer and is a
planned follow-on.

## Examples

### Delaying an irregular feed

```{eval-rst}
.. plotly::
    :include-source: True

    import numpy as np
    import plotly.graph_objects as go
    from screamer import Delay

    idx = np.array([0, 7, 14, 21, 28], dtype=np.int64)
    vals = np.array([1.0, 3.0, 2.0, 4.0, 3.5])
    dv, di = Delay(5)(vals, idx)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=idx, y=vals, mode="lines+markers", name="input"))
    fig.add_trace(go.Scatter(x=di, y=dv, mode="lines+markers", name="Delay(5)"))
    fig.update_layout(title="Delay shifts the index by 5 units",
                      xaxis_title="index", yaxis_title="value",
                      margin=dict(l=20, r=20, t=50, b=20))
    fig.show()
```

<!-- HELP_END -->

## Reference

A delay line from signal processing. Composes with `CombineLatest` to build an
as-of-lagged join, and underpins `forecast_pairs(duration=)`.
```

- [ ] **Step 4: Add the `[Unreleased]` changelog entry** in `CHANGELOG.md` under `### Added`:
```markdown
* `Delay(duration)` stream op: re-stamp each event's index by a time offset (the
  time-based counterpart of `Lag`). Requires an explicit index; lossless, 1:1,
  no warmup.
```

- [ ] **Step 5: Regenerate the registry and build docs**

Run: `poetry run python devtools/build_help_registry.py && poetry run python devtools/build_topic_pages.py && make docs`
Expected: `help.json` gains a `Delay` entry; `make docs` exit 0; the `Delay` page renders (plotly iframe), no new orphan warnings.

- [ ] **Step 6: Commit**

```bash
git add tests/test_streams_delay.py docs/functions_streams/Delay.md screamer/data/help.json CHANGELOG.md docs/by_group docs/by_group_index.rst
git commit -m "docs(streams): Delay page + batch==stream test + changelog"
```

---

## Task 3: `screamer.supervised.forecast_pairs` (count mode)

**Files:**
- Create: `screamer/supervised.py`
- Modify: `screamer/__init__.py`
- Test: `tests/test_supervised_forecast_pairs.py`

**Interfaces:**
- Consumes: `screamer.Lag`.
- Produces: `screamer.supervised.forecast_pairs(X, y, *, count=None, duration=None, dropna=False) -> (X_shifted, y, as_of)`. This task implements `count=` (and the one-of validation shell); `duration=` raises `NotImplementedError` until Task 4.

- [ ] **Step 1: Write the failing tests** (`tests/test_supervised_forecast_pairs.py`)

```python
import numpy as np
import pytest
from screamer.supervised import forecast_pairs


def test_forecast_pairs_count_pairs_features_with_future_target():
    # feature at row t pairs with the target `count` rows later
    X = np.arange(6.0)                       # 0,1,2,3,4,5
    y = np.arange(6.0) * 10                  # 0,10,20,30,40,50
    Xs, ys, as_of = forecast_pairs(X, y, count=2)
    # row t holds feature X[t-2] and target y[t]; first 2 rows warm up to NaN
    assert np.isnan(Xs[:2]).all()
    np.testing.assert_array_equal(Xs[2:], [0.0, 1.0, 2.0, 3.0])   # X[t-2]
    np.testing.assert_array_equal(ys, y)                          # y untouched
    np.testing.assert_array_equal(as_of, np.arange(6))


def test_forecast_pairs_count_dropna_returns_clean_pairs():
    X = np.arange(6.0)
    y = np.arange(6.0) * 10
    Xs, ys, as_of = forecast_pairs(X, y, count=2, dropna=True)
    assert not np.isnan(Xs).any()
    np.testing.assert_array_equal(Xs, [0.0, 1.0, 2.0, 3.0])
    np.testing.assert_array_equal(ys, [20.0, 30.0, 40.0, 50.0])
    np.testing.assert_array_equal(as_of, [2, 3, 4, 5])


def test_forecast_pairs_count_matches_forward_return_reference():
    # forecast_pairs(X, RollingSum(h)(ret), count=h) reproduces the forward-return pairing
    from screamer import RollingSum
    rng = np.random.default_rng(0)
    n, h = 50, 5
    ret = rng.standard_normal(n) * 1e-3
    X = rng.standard_normal(n)
    y = np.asarray(RollingSum(h)(ret))
    Xs, ys, as_of = forecast_pairs(X, y, count=h, dropna=True)
    # reference: X[s] paired with sum(ret[s+1..s+h]) for valid s
    fwd = np.array([ret[s + 1:s + 1 + h].sum() for s in range(n - h)])
    np.testing.assert_allclose(Xs, X[:n - h])
    np.testing.assert_allclose(ys, fwd)


def test_forecast_pairs_requires_exactly_one_of_count_duration():
    X = np.arange(5.0); y = np.arange(5.0)
    with pytest.raises(ValueError):
        forecast_pairs(X, y)                       # neither
    with pytest.raises(ValueError):
        forecast_pairs(X, y, count=1, duration=1)  # both


def test_forecast_pairs_count_2d_features_per_column():
    X = np.column_stack([np.arange(6.0), np.arange(6.0) * 2])
    y = np.arange(6.0)
    Xs, ys, as_of = forecast_pairs(X, y, count=2, dropna=True)
    np.testing.assert_array_equal(Xs, np.column_stack([[0, 1, 2, 3], [0, 2, 4, 6]]))
```

- [ ] **Step 2: Run to verify failure**

Run: `poetry run python -m pytest tests/test_supervised_forecast_pairs.py -q`
Expected: FAIL (`No module named 'screamer.supervised'`).

- [ ] **Step 3: Create `screamer/supervised.py`**

```python
"""Offline supervised-learning helpers built on screamer's causal ops.

forecast_pairs builds a forecasting training set: it lags the features so each row
pairs features from the past with a target realized in their future. The pairing is
causal (it lags X, never leads y), so nothing here peeks into the future; the target
must itself be causal (known as-of its own index), typically a rolling trailing
quantity. These utilities are training-time only.
"""
from __future__ import annotations

import numpy as np

from . import Lag

__all__ = ["forecast_pairs"]


def _leading_nan_mask(a):
    """True where a row is fully finite (a is 1-D or 2-D, per-row over columns)."""
    a = np.asarray(a, dtype=float)
    if a.ndim == 1:
        return np.isfinite(a)
    return np.isfinite(a).all(axis=tuple(range(1, a.ndim)))


def forecast_pairs(X, y, *, count=None, duration=None, dropna=False):
    """Pair features with a target `count` events (or `duration` index-units) ahead.

    Returns (X_shifted, y, as_of). Row t holds the features from `count` events ago
    aligned with the target at t, so a model learns to predict `count` ahead. The
    first `count` rows of X_shifted are NaN (warmup); `dropna=True` drops them.
    `as_of` is each row's completion index (when its target is realized).

    Exactly one of `count` / `duration`. `count` is event-based and needs no index;
    `duration` is time-based (see Delay) and needs an index on X and y.
    """
    if (count is None) == (duration is None):
        raise ValueError("pass exactly one of count= or duration=")
    if duration is not None:
        raise NotImplementedError("duration= mode lands in a later task")

    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(X) != len(y):
        raise ValueError("X and y must share the same length (time axis)")
    Xs = np.asarray(Lag(int(count))(X), dtype=float)
    as_of = np.arange(len(X))
    if dropna:
        keep = _leading_nan_mask(Xs)
        return Xs[keep], y[keep], as_of[keep]
    return Xs, y, as_of
```

- [ ] **Step 4: Export the module** in `screamer/__init__.py`

Add near the other submodule imports so `screamer.supervised` is importable (the module imports `Lag` from the package, so it loads after the bindings):
```python
from . import supervised  # noqa: F401
```
(If `__init__.py` is generated by `make regen-init`, add `supervised` to the generator's hand-maintained submodule list instead; confirm `import screamer.supervised` works after `make regen-init`.)

- [ ] **Step 5: Run to verify pass**

Run: `poetry run python -m pytest tests/test_supervised_forecast_pairs.py -q`
Expected: PASS for the count-mode tests (the both/neither validation, the parity-with-forward-return, 2D per-column, and dropna). The duration test is not present yet.

- [ ] **Step 6: Commit**

```bash
git add screamer/supervised.py screamer/__init__.py tests/test_supervised_forecast_pairs.py
git commit -m "feat(supervised): forecast_pairs count mode (lag features to future target)"
```

---

## Task 4: `forecast_pairs` duration mode (delegates to `Delay` + `CombineLatest`)

**Files:**
- Modify: `screamer/supervised.py`
- Test: `tests/test_supervised_forecast_pairs.py`

**Interfaces:**
- Consumes: `screamer.Delay`, `screamer.CombineLatest`.
- Produces: `forecast_pairs(..., duration=D)` support. X and y are each passed as a `(values, index)` pair; the result pairs each target observation with the feature as-of `(target_index - duration)`.

- [ ] **Step 1: Write the failing tests**

```python
def test_forecast_pairs_duration_matches_count_on_regular_grid():
    # on a regular grid, duration = count * step gives the same pairs as count
    rng = np.random.default_rng(1)
    n, step, h = 40, 100, 3
    idx = np.arange(n, dtype=np.int64) * step
    Xv = rng.standard_normal(n)
    yv = rng.standard_normal(n)
    Xc, yc, ac = forecast_pairs(Xv, yv, count=h, dropna=True)
    Xd, yd, ad = forecast_pairs((Xv, idx), (yv, idx), duration=h * step, dropna=True)
    np.testing.assert_allclose(Xd, Xc)
    np.testing.assert_allclose(yd, yc)


def test_forecast_pairs_duration_requires_index():
    with pytest.raises(TypeError):
        forecast_pairs(np.arange(5.0), np.arange(5.0), duration=2)   # bare arrays


def test_forecast_pairs_duration_async_pairs_by_walltime():
    # X ticks every 10, y ticks every 15; duration 10 pairs y[t] with X as-of (t-10)
    Xv = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    Xi = np.array([0, 10, 20, 30, 40, 50], dtype=np.int64)
    yv = np.array([100.0, 200.0, 300.0])
    yi = np.array([15, 30, 45], dtype=np.int64)
    Xs, ys, as_of = forecast_pairs((Xv, Xi), (yv, yi), duration=10, dropna=True)
    # y at 15 -> X as-of 5 -> X at 0 = 1 ; y at 30 -> X as-of 20 = 3 ; y at 45 -> X as-of 35 -> X at 30 = 4
    np.testing.assert_array_equal(ys, [100.0, 200.0, 300.0])
    np.testing.assert_allclose(Xs, [1.0, 3.0, 4.0])
    np.testing.assert_array_equal(as_of, [15, 30, 45])
```

- [ ] **Step 2: Run to verify failure**

Run: `poetry run python -m pytest tests/test_supervised_forecast_pairs.py -q -k duration`
Expected: FAIL (`NotImplementedError`).

- [ ] **Step 3: Implement duration mode** in `screamer/supervised.py`

Replace the `raise NotImplementedError(...)` with a call to a new helper, and add the helper. Import `Delay` and `CombineLatest` at the top (`from . import Lag, Delay, CombineLatest`). The helper shifts X's index with `Delay`, joins to y with `CombineLatest` (emitting on any event so warmup rows appear), and keeps only the rows on y's clock:

```python
def _forecast_pairs_duration(X, y, duration, dropna):
    if not (isinstance(X, tuple) and isinstance(y, tuple)):
        raise TypeError("duration= mode needs X and y as (values, index) pairs")
    Xv, Xi = np.asarray(X[0], float), np.asarray(X[1])
    yv, yi = np.asarray(y[0], float), np.asarray(y[1])
    if Xv.ndim != 1 or yv.ndim != 1:
        raise ValueError("duration= mode supports 1-D X and y (one feature, one target)")
    Xsv, Xsi = Delay(int(duration))(Xv, Xi)                 # X re-stamped +duration
    combined, cidx = CombineLatest(emit="on_any")((Xsv, Xsi), (yv, yi))
    keep_clock = np.isin(cidx, np.asarray(yi, dtype=cidx.dtype))
    Xs = combined[keep_clock, 0]
    ys = combined[keep_clock, 1]
    as_of = cidx[keep_clock]
    if dropna:
        m = np.isfinite(Xs) & np.isfinite(ys)
        return Xs[m], ys[m], as_of[m]
    return Xs, ys, as_of
```

And in `forecast_pairs`, dispatch:
```python
    if duration is not None:
        return _forecast_pairs_duration(X, y, duration, dropna)
```

- [ ] **Step 4: Run to verify pass**

Run: `poetry run python -m pytest tests/test_supervised_forecast_pairs.py -q`
Expected: PASS (all tests, count and duration). If `CombineLatest`'s column order or index dtype differs, adjust the slice/`isin` dtype to match; the invariant asserted is "one row per y observation, X as-of `(y_index - duration)`".

- [ ] **Step 5: Commit**

```bash
git add screamer/supervised.py tests/test_supervised_forecast_pairs.py
git commit -m "feat(supervised): forecast_pairs duration mode via Delay + CombineLatest"
```

---

## Task 5: `forecast_pairs` docs, help, and full verification

**Files:**
- Create: `docs/functions_supervised/forecast_pairs.md`
- Modify: `docs/topics.yml` (add a `supervised` topic if needed), `CHANGELOG.md`

- [ ] **Step 1: Create the docs page** `docs/functions_supervised/forecast_pairs.md`

Front-matter and body describing the pairing, the causal framing (lags X, never leads y), the `count`/`duration` one-of, the y-must-be-causal contract, and the `as_of` return. Use `topics: [supervised]`, `covers: [forecast_pairs]`. No em-dashes. Include a short runnable example building `y` from `RollingSum` and calling `forecast_pairs(X, y, count=h, dropna=True)`.

- [ ] **Step 2: Register the topic** if `supervised` is not in `docs/topics.yml`: add a `supervised` topic entry (a name + one-line description), mirroring the `streams` entry, so the doc page homes. Confirm the doc-group build maps `functions_supervised/` (if the builder scans `functions_*` dirs, it is automatic; if it enumerates known dirs, add `functions_supervised`).

- [ ] **Step 3: Add the `[Unreleased]` changelog entry** under `### Added`:
```markdown
* `screamer.supervised.forecast_pairs(X, y, count=|duration=)`: build a forecasting
  training set by lagging features to align with a future causal target. Returns
  `(X_shifted, y, as_of)`; `count=` is event-based, `duration=` is time-based (uses
  `Delay`, needs an index). Fully causal (lags X, never leads y).
```

- [ ] **Step 4: Full regen + suite**

Run:
```bash
poetry run python devtools/build_help_registry.py
poetry run python devtools/build_topic_pages.py
make regen-init
poetry run python -m pytest -q
```
Expected: `help.json` gains `forecast_pairs`; the whole suite passes (Delay + forecast_pairs plus everything unchanged).

- [ ] **Step 5: Docs build**

Run: `make docs`
Expected: exit 0; the `Delay` and `forecast_pairs` pages render and home; no new orphan warnings.

- [ ] **Step 6: Commit**

```bash
git add docs/functions_supervised/forecast_pairs.md docs/topics.yml CHANGELOG.md screamer/data/help.json docs/by_group docs/by_group_index.rst
git commit -m "docs(supervised): forecast_pairs page + topic + changelog"
```

---

## Self-Review

**Spec coverage:**
- `Lag` unchanged: no task touches it. Covered.
- `Delay(duration)` C++ stream op, numeric index units, re-stamp, lossless/1:1/no-warmup, index required: Task 1 (node + wiring + surface), Task 2 (docs + batch==stream). Covered.
- `Delay` batch trivial + streaming: the node forwards immediately; batch and `.live()` are proven equal in Task 2. The reorder-buffer (fused-live-merge) case is explicitly out of scope and documented (Task 2 docs Limitations). Covered with the noted deferral.
- `forecast_pairs(count=|duration=)`, one-of, y-causal contract, NaN-warmup + dropna, as_of: Task 3 (count), Task 4 (duration), Task 5 (docs). Covered.
- C++/Python split (Delay C++, forecast_pairs thin Python composing C++ nodes): honored (forecast_pairs calls `Lag`/`Delay`/`CombineLatest`). Covered.
- Naming (Delay, forecast_pairs, duration): as specified. Covered.
- Problems/policies: ties + boundary + tail are `CombineLatest`/engine behaviors exercised by the async and batch==stream tests; the index requirement is tested (Task 1 Step 1, Task 4 Step 1); the buffer-bound and out-of-order policies are not reachable in the delivered direct-call scope (no reorder buffer shipped) and are noted with the deferral. Covered for the delivered scope.
- Migration (screamer-bots forward_* -> forecast_pairs): out of scope per the spec (downstream follow-up); no task. Correct.

**Placeholder scan:** No TBD/TODO. All C++ is exact (from the traced `Resample` recipe). The duration-mode column/dtype adjustment is called out as "match the actual `CombineLatest` output" with the pairing invariant stated, which is guidance for a known-variable API surface, not a placeholder for missing logic.

**Type consistency:** `Delay(duration)` and the `delay()`/`_delay_via_cpp` helpers use `duration` throughout; the graph path is `gb.add_delay(inp, int(kwargs["duration"]))` matching the binding `add_delay(inputs, duration)` matching `GraphBuilder::add_delay(inputs, duration)` matching `DelayNode(duration, downstream)`. `forecast_pairs(X, y, *, count, duration, dropna)` returns `(X_shifted, y, as_of)` consistently across count and duration modes and all tests. `NodeKind::Delay` and `NodeSpec.delay_duration` are used identically in graph.h and compiled_graph.h.

---

## Notes for the implementer

- The C++ tasks are transcription from a traced recipe (the `Resample` node is the exact template). Build with `make install-dev` after every C++ edit; a missing switch case in `compiled_graph.h` shows up as an unhandled `NodeKind` at compile time.
- `Delay` is deliberately stateless: do not add it to `reset_nodes_` and do not give it a params struct beyond the single `duration`.
- `forecast_pairs` must not reimplement the shift in numpy: `count=` goes through `Lag`, `duration=` through `Delay`. That is the whole point (one blessed shift, compute in C++).
- The `duration=` column/index dtype details of `CombineLatest`'s output are the one place to verify against the running API (Task 4 Step 4); assert the pairing invariant, not the internal layout.
