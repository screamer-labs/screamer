# screamer.js Phase 5: DAG/Pipeline Surface - Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Expose screamer's DAG/Pipeline (define-then-bind: `Input`, op composition into nodes, `combineLatest`, `resample`, `select`/`dropna`/`filter`/`delay`, batch + live execution) in the JS binding, parity-faithful to Python, over the pure header-only `dag/` engine.

**Architecture:** A hand-written Embind layer registers the pure `dag::GraphBuilder` and `dag::CompiledGraph` (plus a flat `OutputBuffer` marshaller), porting the nanobind `bindings/bindings_dag.cpp`. A JS layer mirrors Python's two-layer design: a symbolic `Node`/`Input`/`Pipeline` graph (pure JS) that compiles to the C++ `GraphBuilder` then runs via `CompiledGraph.runBatch`/live. `wrapOp` and the combinator factories gain a node branch: given a `Node` argument they compose a node instead of computing.

**Tech Stack:** C++17 + Embind (emcc), the header-only `dag/` engine, TypeScript, Node tests, the Python screamer Pipeline as the parity oracle.

## Global Constraints

- Additive. MUST NOT modify the pure `dag/` engine, any Python binding, operator header, C++ compute code, or the Phase 1-3 layers except the two documented touch points: (a) extend `js/src/runtime.ts` `wrapOp` with a node branch; (b) add DAG registrations to the WASM Embind build. The Python suite stays 6932/2.
- Parity with Python define-then-bind (`tests/test_dag_exec.py` is the reference): `Input("x")`; `Diff(1)(RollingMean(3)(x))` composes nodes; `new Pipeline([x],[y])`; `p(data)` binds at call time and returns `{ values, index }`; multi-output and `live()` mirror Python.
- **Op lifetime:** a compiled graph holds NON-owning `EvalOp*`. The JS `Pipeline` MUST retain references to every op wrapper used in the graph for the graph's lifetime (mirror Python's `op_refs`), and enforce one op instance per node (mirror `_check_stateful_safety`). Disposing an op whose graph is live is a bug the Pipeline must prevent.
- The resample/agg/mode/fill/label int-code tables are ported verbatim from `screamer/dag.py` (`_RESAMPLE_*_CODE`, `_BAR_AGG_FIXED_PLANS`); a test asserts the JS tables match the Python ones so they cannot drift.
- Skip the pull-based lazy `_LazyDriver` path for this phase (it needs Python-iterator sources); batch `runBatch` + `live()` cover the functionality. Log it as a Phase-6+ follow-up.
- No em-dashes; no version file edits.

## The pure engine interface (what to Embind - from bindings_dag.cpp)

`dag::GraphBuilder` (in `include/screamer/dag/graph.h`): `size_t add_input()`, `add_functor(EvalOp*, vector<size_t> inputs)`, `add_combine_latest(inputs, bool when_all, int64_t max_pending)`, `add_dropna(inputs)`, `add_select(inputs, vector<size_t> cols)`, `add_filter(inputs)`, `add_delay(inputs, ...)`, `add_resample(inputs, ResampleParams)`, `set_outputs(vector<size_t>)`, `GraphSpec spec()`. Free `compile(GraphSpec) -> CompiledGraph`.
`dag::CompiledGraph` (`compiled_graph.h`, non-copyable, move-safe): `reset()`, `push_event(size_t inputIdx, int64_t index, double value)`, `push_event_wide(inputIdx, index, const double* values, size_t width)`, `advance(int64_t now)`, `flush()`, `vector<OutputBuffer> drain()`, `vector<OutputBuffer> run_batch(in_indices, in_vals, in_lens, in_widths={})`.
`dag::OutputBuffer { vector<int64_t> indices; vector<double> values; size_t width; }`.
`dag::ResampleParams` (`resample_params.h`): `{ ResampleMode mode; ResampleAgg agg; ResampleLabel label; ResampleFill fill; int64_t width, origin, count, max_age; double threshold; EvalOp* reducer; vector<ResamplePlanEntry> plan; }`.

## File Structure

- Create `wasm/dag_embind.h` + register block - the Embind wrappers `GraphBuilder`, `CompiledGraph`, and a flat marshaller for outputs. Registered from `wasm/generated/bindings_wasm.cpp` via a new `SCREAMER_REGISTER_DAG()` macro call (add the call; keep the op codegen untouched by having the generator emit the macro call, OR add a small hand-written `wasm/dag_bindings.cpp` compiled alongside - prefer the latter to avoid changing the Phase 2 generator).
- Create `wasm/dag_bindings.cpp` - `#include "dag_embind.h"` + `EMSCRIPTEN_BINDINGS(screamer_dag){ SCREAMER_REGISTER_DAG(); }` (a second bindings TU; Embind merges multiple EMSCRIPTEN_BINDINGS blocks).
- Modify: `wasm/CMakeLists.txt` - add `dag_bindings.cpp` to the sources.
- Create: `js/src/node.ts` - the symbolic `Node`/`Input` classes (`isNode=true`), the combinator node builders.
- Create: `js/src/pipeline.ts` - the `Pipeline` class: validate, compile (walk nodes -> GraphBuilder), bind-and-run, `live()`, op-ref retention.
- Create: `js/src/codes.ts` - the resample/agg/mode/fill/label code tables (ported from dag.py) + a JSON dump for the parity gate.
- Modify: `js/src/runtime.ts` - add the node branch to `wrapOp`.
- Modify: `js/src/index.ts` - export `Input`, `Pipeline`, `combineLatest`, `resample`, `select`, `dropna`, `filter`, `delay`, `merge`.
- Create: `js/src/generated/combinators.ts` (or hand-written) - the combinator factories.
- Create: `devtools/wasm/gen_dag_oracle.py` - Python Pipeline oracle for parity.
- Create: `js/test/pipeline.test.ts`, `js/test/combinators.test.ts` - the JS DAG suite.
- Create: `tests/test_dag_codes_sync.py` - asserts the JS code tables equal the Python ones.

---

### Task 1: DAG Embind layer (GraphBuilder + CompiledGraph + output marshaller)

**Files:**
- Create: `wasm/dag_embind.h`, `wasm/dag_bindings.cpp`
- Modify: `wasm/CMakeLists.txt`
- Test: extend `wasm/smoke/smoke.mjs` or a new `wasm/smoke/dag_smoke.mjs`

**Interfaces:**
- Produces (JS-visible on the module `M`): `GraphBuilder` with `addInput()`, `addFunctor(evalOpPtr, VectorSizeT)`, `addCombineLatest(VectorSizeT, bool, number)`, `addDropna`, `addSelect`, `addFilter`, `addDelay`, `addResample(VectorSizeT, modeCode, aggCode, labelCode, fillCode, width, origin, count, threshold, maxAge, planPtr, planLen, reducerPtr)`, `setOutputs(VectorSizeT)`, `compile(): CompiledGraph`. `CompiledGraph` with `reset/pushEvent/pushEventWide/advance/flush` and a flat marshaller `runBatchFlat(...) -> OutBufFlat` + `drainFlat() -> OutBufFlat` where `OutBufFlat` is a `value_object {indexPtr, valuePtr, rows, width}` (copy into heap buffers the caller frees), mirroring `marshal_gather` in bindings_dag.cpp.

- [ ] **Step 1: Write `wasm/dag_embind.h`**

Wrap `dag::GraphBuilder` and `dag::CompiledGraph`. Take `EvalOp*` as `uintptr_t` from JS (reinterpret, like `evalInto`). For outputs, add a marshaller: given the `vector<OutputBuffer>` from `run_batch`/`drain`, for the (single, or per-output) buffer copy `indices` (as `double` or via a parallel int64 heap - simplest: copy int64 indices into a `double`-typed heap buffer as BigInt is awkward; instead expose indices as a separate `allocI64`/view OR return them as doubles since screamer indices fit in double for realistic sizes - DECISION: return indices as `Float64Array` via a `double` copy, matching the Python `(values, index)` float convention where index is commonly an arange or timestamp; document the >2^53 caveat). Return `{ indexPtr, valuePtr, rows, width }`; JS reads via `viewF64` and frees via `freeBuf`. Provide `SCREAMER_REGISTER_DAG()` registering both classes + `register_vector<size_t>` (as `VectorSizeT`) + the `OutBufFlat` value_object.

- [ ] **Step 2: `wasm/dag_bindings.cpp`**

```cpp
#include "dag_embind.h"
EMSCRIPTEN_BINDINGS(screamer_dag) { SCREAMER_REGISTER_DAG(); }
```

- [ ] **Step 3: Add to the build**

In `wasm/CMakeLists.txt`, add `${CMAKE_SOURCE_DIR}/wasm/dag_bindings.cpp` to the target sources (alongside `wasm/generated/bindings_wasm.cpp`).

- [ ] **Step 4: Raw smoke - build a trivial graph in JS**

`wasm/smoke/dag_smoke.mjs`: `const M = await init(); const gb = new M.GraphBuilder(); const x = gb.addInput(); const rm = new M.RollingMean(3, "strict"); const ids = new M.VectorSizeT(); ids.push_back(x); const y = gb.addFunctor(rm.$$?.ptr ?? getPtr(rm), ids); const outs = new M.VectorSizeT(); outs.push_back(y); gb.setOutputs(outs); const cg = gb.compile();` then feed `[1,2,3,4,5]` via `runBatchFlat` and assert the output values equal `[NaN,NaN,2,3,4]` (RollingMean(3)). (Getting the raw `EvalOp*` from an Embind object: expose a helper `opPtr(op): uintptr_t` in `embind_runtime.h` returning `reinterpret_cast<uintptr_t>(&op)`, since the JS wrapper holds the C++ object - add that helper.)

Run: `bash wasm/build-wasm.sh && node wasm/smoke/dag_smoke.mjs`
Expected: prints the RollingMean output and asserts it matches `[NaN,NaN,2,3,4]`; `DAG SMOKE OK`. A build/link error or wrong output is a real finding.

- [ ] **Step 5: Commit**

```bash
git add wasm/dag_embind.h wasm/dag_bindings.cpp wasm/CMakeLists.txt wasm/smoke/dag_smoke.mjs
git commit -m "feat(wasm): Embind DAG layer (GraphBuilder + CompiledGraph + flat output marshaller)"
```

---

### Task 2: JS Node/Pipeline + wrapOp node branch

**Files:**
- Create: `js/src/node.ts`, `js/src/pipeline.ts`
- Modify: `js/src/runtime.ts` (node branch), `js/src/index.ts`
- Test: `js/test/pipeline.test.ts`

**Interfaces:**
- `Node { readonly isNode = true; op: unknown; inputs: Node[]; }`; `Input(name: string): Node`.
- `wrapOp`: at the top of `call`, if any arg `isNode`, return `new Node(this-op-wrapper, args)` (defer). The op wrapper must be retained by the eventual Pipeline.
- `Pipeline(inputs: Node[], outputs: Node[], opts?)`: `.compile()` walks the node DAG (topo), calls `GraphBuilder.addInput/addFunctor/...`, `setOutputs`, `compile()`, retaining every op wrapper; `pipeline(feeds)` binds data and returns `{ values: Float64Array | NdArray, index: Float64Array }`; `.dispose()` frees the CompiledGraph and releases op refs.

- [ ] **Step 1: `js/src/node.ts`**

```ts
export class Node { readonly isNode = true as const;
  constructor(public op: unknown, public inputs: Node[]) {} }
export function Input(name: string): Node { return new Node({ input: name }, []); }
export function isNode(x: unknown): x is Node { return !!x && (x as any).isNode === true; }
```

- [ ] **Step 2: node branch in `wrapOp`**

In `js/src/runtime.ts`, at the very top of the `call` dispatcher, before the numeric checks: `if (args.some(isNode)) { return new Node({ functor: raw, wrapper: op }, args.filter(isNode)); }` (retain both the raw op and the wrapper so the Pipeline can pin lifetime and get the `EvalOp*`). Import `Node`/`isNode` from `./node.js`. This must not disturb the existing 42 tests (no arg is a Node in those).

- [ ] **Step 3: `js/src/pipeline.ts` - compile + run**

Walk outputs' transitive inputs, assign ids, enforce one-functor-per-node and all-inputs-used (port `_check_stateful_safety` + reachability from `dag.py`). For each node: Input -> `addInput`; functor node -> materialize a `VectorSizeT` of input ids, `addFunctor(opPtr(raw), ids)`, and push the op wrapper into `this.ops` (retention). Operator nodes (combinators) dispatch by kind (Task 3). `setOutputs`, `compile()`. `pipeline(feeds)`: normalize each feed to `{values: Float64Array, index: Float64Array}` (bare array -> index = 0..n-1), marshal into heap buffers, call `runBatchFlat`, read `OutBufFlat` via `viewF64`, copy out, `freeBuf`. Return `{values, index}` (single output) or per-output. `dispose()` frees the graph then the retained op wrappers.

- [ ] **Step 4: Test a functor-only pipeline vs the point-op result**

`pipeline.test.ts`: `await ready(); const x = Input("x"); const y = Diff(1)(RollingMean(3)(x)); const p = new Pipeline([x],[y]); const {values} = p(new Float64Array([1,2,3,4,5,6])); ` assert `values` equals the direct `Diff(1)(RollingMean(3)(...))` composed by hand on the same data (compute the point-op way and compare). Dispose.

Run: `cd js && npm run build:wasm && node --import tsx --test test/pipeline.test.ts`
Expected: pass. A mismatch means the graph execution diverges from the point-op path - diagnose.

- [ ] **Step 5: Commit**

```bash
git add js/src/node.ts js/src/pipeline.ts js/src/runtime.ts js/src/index.ts js/test/pipeline.test.ts
git commit -m "feat(js): symbolic Node/Pipeline + wrapOp node branch (define-then-bind)"
```

---

### Task 3: Combinators + resample code tables

**Files:**
- Create: `js/src/codes.ts`, `js/src/combinators.ts`
- Modify: `js/src/pipeline.ts` (compile the operator-node kinds), `js/src/index.ts`
- Test: `js/test/combinators.test.ts`, `tests/test_dag_codes_sync.py`

**Interfaces:**
- `combineLatest(nodes: Node[], opts?: {emit?: "when_all"|"on_any", maxPending?: number}): Node`; `resample(node, opts): Node`; `select(node, cols): Node`; `dropna(...nodes): Node`; `filter(...): Node`; `delay(node, k): Node`; `merge(...nodes): Node`. Each returns a `Node` whose `op` encodes the kind + params.
- `codes.ts`: `RESAMPLE_MODE_CODE`, `RESAMPLE_AGG_CODE`, `RESAMPLE_FILL_CODE`, `RESAMPLE_LABEL_CODE`, `BAR_AGG_FIXED_PLANS` - copied verbatim from `screamer/dag.py`.

- [ ] **Step 1: Port the code tables**

Copy the `_RESAMPLE_MODE_CODE`, `_RESAMPLE_AGG_CODE`, `_RESAMPLE_FILL_CODE`, `_RESAMPLE_LABEL_CODE`, `_BAR_AGG_FIXED_PLANS` dicts from `screamer/dag.py` into `js/src/codes.ts` as objects, plus a `--dump` path (a small script or an exported const) so the parity test can compare.

- [ ] **Step 2: Combinator factories**

`combinators.ts`: each returns `new Node({ combinator: "combineLatest"|"resample"|..., params }, nodeArgs)`. `resample` translates its opts to codes via `codes.ts`.

- [ ] **Step 3: Compile operator nodes in `pipeline.ts`**

Extend the node walker: `combineLatest` node -> `addCombineLatest(ids, whenAll, maxPending)`; `resample` -> build `ResampleParams` fields and `addResample(ids, modeCode, aggCode, labelCode, fillCode, width, origin, count, threshold, maxAge, planPtr, planLen, reducerPtr)`; `select`/`dropna`/`filter`/`delay` similarly.

- [ ] **Step 4: Code-sync gate**

`tests/test_dag_codes_sync.py`: import the Python `dag.py` code dicts and assert they equal the JS `codes.ts` tables (parse the TS objects or a JSON the `--dump` emits). Fails if either drifts.

- [ ] **Step 5: Combinator parity test**

`combinators.test.ts`: build a `combineLatest` of two inputs feeding a functor, and a `resample(x, {mode:"by_index", agg:"mean", every:5})`, run in JS; assert against expected values (use small hand-checked inputs, or the Task-4 oracle if it lands first).

Run: `cd js && npm run build:wasm && node --import tsx --test test/combinators.test.ts` and `poetry run pytest tests/test_dag_codes_sync.py -q`
Expected: both pass.

- [ ] **Step 6: Commit**

```bash
git add js/src/codes.ts js/src/combinators.ts js/src/pipeline.ts js/src/index.ts js/test/combinators.test.ts tests/test_dag_codes_sync.py
git commit -m "feat(js): combinators (combineLatest/resample/select/...) + code-table parity gate"
```

---

### Task 4: Pipeline parity vs the Python oracle + live streaming

**Files:**
- Create: `devtools/wasm/gen_dag_oracle.py`, `wasm/smoke/dag_oracle.json`
- Modify: `js/src/pipeline.ts` (add `.live()`)
- Test: `js/test/dag_parity.test.ts`, `js/test/live.test.ts`

**Interfaces:**
- `pipeline.live(): { push(name, index, value), advance(now), flush(), result() }` over `CompiledGraph.pushEvent/advance/flush/drain`.

- [ ] **Step 1: Oracle generator**

`gen_dag_oracle.py` (`poetry run python`): build ~8 representative Pipelines in Python (functor chain; combine_latest -> functor; resample by_index/by_count/mean/ohlc; select; dropna; delay) on fixed inputs, record `{name, graph_spec (a JS-buildable description), feeds, expect_values, expect_index}`. Emit `wasm/smoke/dag_oracle.json`.

- [ ] **Step 2: `.live()` on Pipeline**

Add the live driver wrapping `pushEvent`/`advance`/`flush`/`drain`.

- [ ] **Step 3: Parity + live tests**

`dag_parity.test.ts`: for each oracle entry, build the equivalent JS Pipeline (a small builder that reads the graph description), run batch, assert values+index match within 1e-9. `live.test.ts`: drive one pipeline event-by-event via `.live()` and assert it equals the batch result (the streaming==batch invariant).

Run: `cd js && npm run build:wasm && poetry run python devtools/wasm/gen_dag_oracle.py && node --import tsx --test test/dag_parity.test.ts test/live.test.ts`
Expected: all pass. A mismatch is a real finding (diagnose graph-build vs execution).

- [ ] **Step 4: Commit**

```bash
git add devtools/wasm/gen_dag_oracle.py wasm/smoke/dag_oracle.json js/src/pipeline.ts js/test/dag_parity.test.ts js/test/live.test.ts
git commit -m "test(js): Pipeline parity vs Python oracle + live streaming (batch==stream)"
```

---

### Task 5: Lifetime safety + full-suite integration

**Files:**
- Test: `js/test/dag_lifetime.test.ts`
- Modify: `tests/test_ts_api_fresh.py` or a note (confirm no Python behavior change)

**Interfaces:** none new.

- [ ] **Step 1: Lifetime tests**

`dag_lifetime.test.ts`: assert reusing one op instance across two nodes throws (the one-functor-per-node rule); assert disposing an op used by a live Pipeline is prevented or the Pipeline retains it (construct a pipeline, drop the op reference, force GC if possible, run - must still work because the Pipeline pinned it); assert `Pipeline.dispose()` frees the graph and does not double-free the ops; assert an input never fed throws a clear error at call time.

- [ ] **Step 2: Full JS suite + Python suite**

Run: `cd js && npm run build:wasm && node --import tsx --test test/*.test.ts` -> all pass (the ~42 point-op tests + the DAG tests).
Run: `poetry run pytest -q 2>&1 | tail -3` -> 6933 passed, 2 skipped (6932 + the code-sync test; adjust if you added more), confirming no Python behavior change.

- [ ] **Step 3: Confirm the WASM size delta**

Run: `bash wasm/build-wasm.sh` and note the new `screamer.wasm` gzip size (the DAG layer adds to the ~85 KB - report the delta; if it pushes well past 90 KB, note it for a Phase-6 size pass, do not block).

- [ ] **Step 4: Commit**

```bash
git add js/test/dag_lifetime.test.ts
git commit -m "test(js): DAG op-lifetime safety + full-suite integration"
```

---

## Notes for the executor

- Task 1 is the crux (the Embind port + the output marshaller + getting the raw `EvalOp*` from a JS op). The `opPtr` helper and the flat `OutBufFlat` marshaller are the two idioms to nail; `bindings/bindings_dag.cpp` is the exact reference for the wrapper semantics.
- The op-lifetime rule is the top correctness risk: the graph holds non-owning `EvalOp*`, so the JS `Pipeline` MUST retain the op wrappers and forbid disposing them while the graph lives. Task 5 tests this explicitly.
- Index-as-Float64 loses precision above 2^53; acceptable for arange/typical timestamps, documented. A BigInt64 index path is a possible later refinement.
- Skip the pull-based lazy `_LazyDriver`; batch + live cover parity. Log it.
- Everything is additive except the two documented touch points (`wrapOp` node branch; the WASM build's DAG source). No change to the pure engine, Python, or the Phase 2 op codegen.
