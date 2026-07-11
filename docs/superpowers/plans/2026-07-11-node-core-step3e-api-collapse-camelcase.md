# Node core step 3E: collapse the API to `Op(config)(data)` (CamelCase)

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Collapse the two API shapes into one. Today math ops are functor classes `Op(cfg)(data)` and stream ops are functions `op(data, cfg)`. Make the stream operators CamelCase config-classes too: `resample`->`Resample`, `dropna`->`Dropna`, `select`->`Select`, `combine_latest`->`CombineLatest`, `merge`->`Merge` (`Filter` already done in 3A). Remove the transitional `every=` from resample (Option B: `freq`/`count`). Migrate all call sites, tests, and docs. This delivers user goal #2.

**Scope decisions (settled):**
- **Keep `Stream`** (the return container) - it carries column labels (ohlc) a bare tuple loses. Only the CALL shape changes; return types are unchanged. Full `Stream` removal is a SEPARATE later decision, not in this plan.
- No lowercase aliases in the END state ("legacy removed"). Transitional shims exist ONLY during migration and are deleted in the final task.
- `replay`/`split` stay functions (drivers/utilities, not stream operators).

**Architecture:** Each stream op becomes a class whose `__init__` takes the config and whose `__call__(data, index=None)` (or `__call__(*data)` for the variadic joins) runs the current function body. The Dag compile map (`dag.py` `_compile_cpp`) dispatches on the operator identity `getattr(fn, "__name__", "")`; it is updated to the CamelCase class names (mirroring how `Filter` already registers as `name == "Filter"`). Migration keeps the suite green by adding the classes first (with temporary lowercase shims), migrating callers in batches, then deleting the shims + `every=` last.

**Tech Stack:** Python (`screamer/streams.py`, `screamer/dag.py`, `screamer/__init__.py` via `devtools/generate_screamer__init__.py`), pytest. `make install-dev` only if a binding/C++ file is touched (not expected).

## Global Constraints

- **Behavior-preserving:** the classes produce byte-identical output to the functions they replace; `batch == lazy == graph` unchanged. Suite green after each task (baseline: 4185 passed, 2 skipped, 0 failed).
- Config in the constructor; data in the call. `Op(config)(data)`. Multi-input joins are variadic: `CombineLatest(emit=...)(a, b, c)`, `Merge()(a, b)`.
- Remove `every=` from the public `Resample` (freq/count only); the internal window param may stay private.
- CamelCase class names match the existing functor convention (`Asin`, `RollingMean`). Export via the generator.
- No version-file edits. No em-dashes, no ` -- `. Preserve every existing error message and semantic.

## Target signatures

- `Resample(freq=None, count=None, agg="last", origin=0, label="left", fill="skip")(values, index=None)`
- `Dropna(how="any")(values, index=None)`
- `Select(columns)(values, index=None)`
- `CombineLatest(emit="when_all")(*values, index=None)` (drop `func=`; compose a functor on the output)
- `Merge()(*values, index=None)`

---

### Task 1: introduce the five CamelCase classes (with transitional lowercase shims)

**Files:** `screamer/streams.py` (add classes; rename current function bodies to private `_<op>_impl` or keep functions as the impl and have classes delegate), `screamer/dag.py` (compile map -> CamelCase names), `screamer/__init__.py` (regen exports). Test: `tests/test_op_config_shape.py` (equivalence: `Resample(cfg)(data) == resample(data, cfg)` for each op, across batch/lazy/graph).

- [ ] **Step 1:** for each op, add a class `Resample`/`Dropna`/`Select`/`CombineLatest`/`Merge` whose `__init__` stores the config and whose `__call__` delegates to the existing function body (data-first). Give each a `__name__` matching the class (for the compile map). Drop `func=` from `CombineLatest`; drop `every=` from `Resample` (map freq/count through the existing translation).
- [ ] **Step 2:** update `dag.py` `_compile_cpp` so the operator-node identities are the class names: `name == "Resample"`, `"Dropna"`, `"Select"`, `"CombineLatest"` (and the node-building `make_operator_node(<Class>, ...)` inside each class's `is_node` branch passes the CLASS). `Merge` raises on Node inputs (unchanged). Verify a graph built with the classes compiles (e.g. `Dropna()(CombineLatest()(a, b))`).
- [ ] **Step 3:** keep the lowercase `resample`/`dropna`/`select`/`combine_latest`/`merge` as THIN shims delegating to the classes (so existing tests pass during migration). Mark them clearly as transitional (a comment: removed in the final 3E task).
- [ ] **Step 4:** export the classes (regen `screamer/__init__.py` via `PYTHONPATH=. python devtools/generate_screamer__init__.py`; confirm `from screamer import Resample, Dropna, Select, CombineLatest, Merge`).
- [ ] **Step 5:** `tests/test_op_config_shape.py` - for each op, assert the class form equals the (still-present) function form byte-for-byte across batch, lazy, graph. Full suite green (functions still work via shims).
- [ ] **Step 6: commit.**

---

### Tasks 2-K: migrate call sites (tests + docs) to the classes, in batches

Each task takes a BATCH of files (grouped so the suite can run green after each) and rewrites `op(DATA, INDEX, **cfg)` -> `Op(**cfg)(DATA, INDEX)` for the five ops. The lowercase shims still exist, so a partially-migrated tree stays green; each task fully migrates its batch and runs the suite.

- [ ] **Batch A - resample tests:** `tests/test_streams_resample.py`, `test_resample_*.py`, `test_multi_resample*.py`, `test_dag_resample.py`, `test_resample_advance.py`. Rewrite `resample(...)` -> `Resample(...)(...)` and `multi_resample(...)` similarly (or its class if introduced). Suite green.
- [ ] **Batch B - dropna/select/filter tests:** `test_dropna_node_backing.py`, `test_select_node_backing.py`, `test_dag_dropna.py`, `test_stream_columns.py`, `test_streams_select.py`, and any `dropna`/`select` uses elsewhere. Suite green.
- [ ] **Batch C - combine_latest/merge tests:** `test_combine_latest*.py`, `test_merge*.py`, `test_streams_*.py`, `test_dag_io.py`, `test_replay*.py` (replay stays a function but its merge inputs migrate). Suite green.
- [ ] **Batch D - remaining scattered call sites:** grep `resample(|dropna(|select(|combine_latest(|merge(` across `tests/` and `screamer/` (excluding the shims + class internals) and migrate stragglers. Suite green.
- [ ] **Batch E - docs + notebooks:** `docs/**/*.md`, `docs/notebooks/*.ipynb`, `docs/functions_streams/*` (rename pages `resample.md`->`Resample.md` etc.), regen `help.json`/topic pages. `make notebooks` green.

(Each batch = its own task + review; briefs generated per batch. Keep batches file-scoped so a reviewer can gate one without the others.)

---

### Task Final: delete the lowercase shims + `every=`; end-state clean

**Files:** `screamer/streams.py` (delete the 5 shims + any `every=` remnant), `screamer/__init__.py` (regen), docs.

- [ ] **Step 1:** grep the WHOLE tree (excluding `docs/_build`, `docs/superpowers`) for `resample(`/`dropna(`/`select(`/`combine_latest(`/`merge(` and `every=`. Zero non-class, non-shim hits expected (all migrated).
- [ ] **Step 2:** delete the 5 lowercase shims and remove `every=` entirely (public + any now-unused translation). Confirm the `multi_resample` story (migrate or class-ify consistently).
- [ ] **Step 3:** regen exports/help.json; confirm `screamer.resample` etc. no longer exist and the classes are the sole API.
- [ ] **Step 4:** full suite + `make notebooks` green.
- [ ] **Step 5: commit.**

---

## Self-review notes

- **Suite green throughout:** classes-with-shims (Task 1) -> batch migration (green after each) -> delete shims last. No red window.
- **Behavior-preserving:** the classes delegate to the SAME bodies; `test_op_config_shape.py` pins class == function byte-for-byte before migration; the existing per-op suites pin behavior after.
- **`Stream` kept** (column labels preserved); this is the CALL-shape collapse only.
- **Compile-map identity** switches to CamelCase names (like `Filter`); verify graph composition of the classes.
- **Do NOT** delete `Stream`, change return types, or alter operator semantics - only the call shape + `every=` removal.
- **Follow-up carried in:** factor the shared dropna/select tee-demux/2D-Dag scaffolding (3C deferral) while touching streams.py, IF clean.
