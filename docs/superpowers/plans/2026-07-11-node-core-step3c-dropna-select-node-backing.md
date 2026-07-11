# Node core step 3C: route Dropna/Select through the C++ nodes

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Remove the numpy-eager and Python-generator implementations of `dropna`/`select` and route every regime through the existing C++ `DropNaNode`/`SelectNode` via the `Dag` machinery. This kills the duplicate implementations the user flagged (numpy eager vs C++ graph node that can diverge, unreachable from a non-Python binding). Behavior-preserving: byte-identical output.

**Architecture (verified):** The C++ `DropNaNode`/`SelectNode` already exist and work as graph nodes on multi-column frames. A standalone `dropna(1d)` routes through a one-input `Dag(Input -> dropna_node)`; a standalone `dropna(2d)` routes through `Dag(cols -> combine_latest(*cols) -> dropna_node)` where `combine_latest` positional IS the column pack (VERIFIED: reproduces numpy `dropna(2d)` byte-for-byte for how=any AND how=all). Same for `select`. This reuses the `Dag` call machinery exactly like `Filter` (step 3A) and `resample` (`_resample_via_cpp`) - no engine change, no multi-column-input feature needed.

**Tech Stack:** Python (`screamer/streams.py`), pytest. Uses the existing C++ nodes (no C++ changes expected). `make install-dev` only if a C++/binding file is touched.

## Global Constraints

- **Behavior-preserving, byte-identical.** `batch == lazy == graph`. Every rerouted path asserts equality to the pre-change numpy/generator output. **Use `equal_nan=True`** in array comparisons - `dropna(how="all")` and `select` outputs retain NaN values, and `np.array_equal` treats NaN as unequal by default (this caused a false alarm during planning).
- Suite green after each task (main baseline: 4148 passed, 2 skipped, 0 failed).
- The Python `dropna`/`select` must contain NO compute (no `np.isnan` mask, no column pick loop) after this - only routing to the `Dag`. The gate/pick logic lives solely in the C++ nodes.
- Preserve the exact public API shape for now (`dropna(values, index=None, how="any")`, `select(values, columns, index=None)` returning the same `Stream`/tuple/lazy forms). CamelCase + `Stream` removal is step 3E - do NOT do it here.
- No version-file edits. No em-dashes, no ` -- `. No new numpy compute.

## Interfaces

- Consumes: the existing `combine_latest`, `Input`, `Dag`, `make_operator_node`, and the `dropna`/`select` graph-node path (the `is_node(values)` branch already builds `DropNaNode`/`SelectNode`).
- Produces: `dropna`/`select` whose raw/lazy paths delegate to a `Dag`, with `_dropna_lazy`/`_select_lazy` and the numpy mask/pick deleted.

---

### Task 1: route `dropna` through `DropNaNode`; delete the numpy + generator paths

**Files:** `screamer/streams.py` (`dropna`, `_dropna_lazy`). Test: `tests/test_dag_dropna.py` / a new `tests/test_dropna_node_backing.py`.

- [ ] **Step 1:** capture the oracle from the CURRENT numpy `dropna` before changing it: for 1-D and 2-D arrays, positional and indexed, how="any" and how="all", the survivors and indices. Include an all-NaN row (how="all" drops it), partial-NaN rows, a leading/trailing NaN, and a lazy feed (1-D and 2-D `(value, index)` events). Store expected outputs; assert with `equal_nan=True`.
- [ ] **Step 2:** implement a helper `_dropna_via_cpp(feed_or_feeds, how, ...)` (mirror `_resample_via_cpp`): for a 1-D stream build `Dag([Input], [dropna(Input, how=how)])` and call `dag(feed)`; for a 2-D stream split into N columns and build `Dag([c0..cN], [dropna(combine_latest(c0..cN), how=how)])` and call `dag(*col_feeds)`. Rewrite `dropna`'s raw/Stream branch to delegate to it. Return the SAME shape the old code returned (match `_adapt`'s current output: Stream for the stream regime, `(values, index)` tuple for raw - read what the current `dropna` returns and preserve it exactly).
- [ ] **Step 3:** the lazy path: 1-D lazy delegates to the Dag lazy path (`dag(iter_of_events)`); for 2-D lazy, demux the row-iterator into N column-iterators (e.g. `itertools.tee` + per-column extraction, read in lockstep so tee buffering stays O(1)) and feed the N-input pack Dag. If 2-D lazy through the Dag proves intractable, KEEP a minimal `_dropna_lazy` for the 2-D lazy case ONLY and report it (do not silently keep the numpy eager path - that is the primary duplicate to remove). Delete `_dropna_lazy` if fully replaced.
- [ ] **Step 4:** delete the numpy mask (`np.isnan(...)` + `~mask`) eager compute from `dropna`. Confirm no `np.isnan`/mask compute remains in `dropna`.
- [ ] **Step 5:** assert byte-identical to the Step 1 oracle across all cases (equal_nan=True); confirm `batch == lazy == graph`; full suite green.
- [ ] **Step 6: commit.**

---

### Task 2: route `select` through `SelectNode`; delete the numpy + generator paths

**Files:** `screamer/streams.py` (`select`, `_select_lazy`, and any shared helper from Task 1). Test: existing select tests + a new `tests/test_select_node_backing.py`.

**Interfaces consumed:** Task 1's `_dropna_via_cpp` pattern; generalize the shared 1-D/2-D pack+Dag helper if clean.

- [ ] **Step 1:** oracle from the current numpy `select`: pick single column (returns 1-D), multiple columns (returns 2-D projected), positional + indexed, out-of-range column raises, and lazy (1-D/2-D `(value, index)` events). Capture expected; assert equal_nan=True.
- [ ] **Step 2:** route `select` eager through `SelectNode`: `select(1d)` is a passthrough/identity (or column 0) per current semantics - match it; `select(2d, columns)` builds `Dag([c0..cN], [select(combine_latest(c0..cN), columns=columns)])` and calls it. Preserve the current return shape (single-column -> 1-D; multi -> 2-D). Keep the out-of-range column validation (where does it live now - Python or the node? preserve the same error).
- [ ] **Step 3:** the lazy path through the Dag (1-D direct; 2-D via the Task 1 demux). Delete `_select_lazy` if fully replaced (else keep the 2-D-lazy-only case and report).
- [ ] **Step 4:** delete the numpy column-pick compute from `select`.
- [ ] **Step 5:** assert byte-identical (equal_nan=True), `batch == lazy == graph`, full suite green.
- [ ] **Step 6: commit.**

---

## Self-review notes

- **No Python compute left:** after this, `dropna`/`select` route to the `Dag` and contain no `np.isnan`/mask/column-pick. The one allowed exception is a minimal 2-D-lazy shim IF the tee-demux is intractable (must be reported, not silent).
- **equal_nan=True** in every array oracle (how="all"/select outputs retain NaN).
- **Return shape preserved** exactly (Stream vs tuple per regime) - this is NOT the Stream-removal step (3E). Read what the current code returns and match it.
- **Reuse the Dag machinery** (like Filter/resample); do not write a bespoke driver. `combine_latest` positional is the verified column pack.
- **Do NOT** rename to CamelCase, remove `every=`, touch `Stream`, or change the public signature here - those are step 3E.
