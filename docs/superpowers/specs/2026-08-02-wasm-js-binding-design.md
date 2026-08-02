# screamer.js: JavaScript/WASM Binding Design

**Status:** design approved, ready for an implementation plan. Feasibility proven by the Emscripten spike on 2026-08-02 (`docs/superpowers/spikes/2026-08-02-wasm/REPORT.md`), which compiled the real `detail::RollingMean` kernel to WASM, ran it under Node, and matched Python `screamer.RollingMean(3)` bit for bit.

**Goal:** a full-parity JavaScript binding of screamer, `@screamer-labs/screamer` on npm, exposing every operator, all four input regimes (scalar, batch, lazy, async), and the Pipeline/DAG engine, backed by the same C++ core the Python package uses.

**Architecture:** three layers. One pure-compute C++ core (no binding library) shared by two thin dispatch wrappers, nanobind for Python and Embind for JavaScript. Compute lives in C++; each binding is a thin driver.

**Tech stack:** C++17, Emscripten + Embind, TypeScript, an op manifest with code generation, golden-vector parity tests against the Python oracle.

## Scope and the decisions behind it

- **Full parity.** All ~248 registry entries (operators, `Pipeline`, combinators, `backtest_report`), all input regimes, published to npm as the definitive JS build.
- **Clean unified split.** One pure-compute base that both bindings share. The polymorphic dispatch relocates into a nanobind wrapper (Python) and an Embind wrapper (JS). Python behavior stays identical, verified by the 6926-test suite.
- **DAG in the first release.** The Pipeline/combinator/resample surface ships with the point operators, not as a fast-follow.
- **npm name `@screamer-labs/screamer`**, matching the GitHub org.
- **Shared versioning.** screamer.js tracks the Python version. The first npm release is 2.x, one version bump drives both, and the npm version reads from the same single source the Python package uses. Version files are never hand-edited; only `make patch/minor/major` moves them.

## Layer 1: the pure C++ core and the base-class split

Today `ScreamerBase` and `FunctorBase` mix two concerns: pure compute and nanobind dispatch. The split separates them so an operator header compiles with no binding library present, which is the WASM prerequisite.

**`ScreamerBase` (1-in/1-out operators) splits cleanly.** Its compute contract is already pure double-in/double-out: `process_scalar`, `process_array_no_stride`, `process_array_stride`, `eval`, `reset` (`base.h:222-240`). Only `operator()(nb::object)` and `process_python_array(nb::ndarray)` are nanobind, and they lift out into the wrapper.

**`FunctorBase` (N-in/M-out operators) needs real extraction, not a file move.** Its pure compute is only `call()`, `process_columns()`, and `eval()`. The batch array loops are inlined inside the nanobind `handle_input_*_numpy` template methods (`functor_base.h:172-300`), which take `nb::ndarray` and return `nb::object` via `make_owned_array`. Unlike `ScreamerBase`, no pure batch method exists to share. The clean split extracts the four batch loops (1i1o, Ni1o, 1iMo, NiMo) into pure `double*` kernels, then both wrappers call them. This is the bulk of the Phase 1 C++ work and the main parity-risk surface, netted by the 6926 tests.

**The `detail::` namespace splits too.** Pure helpers (`TupleOfDoubles`, `write_tuple_to_memory`, used by the pure `eval` at `functor_base.h:164`) move to a pure detail header. The nanobind helpers (`read_contig_double`, `make_owned_array`, `coerce_*`, `load_elem`, `read_n_arrays`) move to a nanobind detail header.

**What is already WASM-ready and reused unchanged:** the `detail/` kernels, `EvalOp`, and the entire `dag/` engine (`Sink`, `Frame`, watermark). The spike confirmed they compile under emcc. No threads, RTTI, SIMD intrinsics, Eigen, Boost, or filesystem appear in the core.

**Dispatch relocation touches the bindings.** `operator()` and `handle_input` are base methods bound via `.def("__call__", &Op::operator())`. Relocating them rewires the 238 registrations to call a dispatch entry that takes a base reference. This is mechanical and code-generation owned. The self-reference, lazy-iterator, and dag-node idioms (`nb::find`, `LazyEvalIterator`, `make_dag_functor_node`) stay entirely in the nanobind wrapper; the Embind side gets JS-native equivalents, so there is no shared code there and no shared-correctness risk.

## Layer 2: the Embind binding and code generation

**The Embind runtime is arity-agnostic and written once.** `EvalOp::eval(const double* in, double* out)` is already the uniform N-to-M contract every operator implements, so the JS-facing runtime binds once on the `EvalOp` base:

- `evalInto(inPtr, outPtr)` calls `this->eval(...)` over heap pointers, as the spike proved.
- `evalBatchInto(inPtr, outPtr, rows)` loops `eval` over rows.
- `reset()`, `nIn()`, `nOut()`.

Each operator's entire registration is then its constructor:

```cpp
class_<RollingMean, base<EvalOp>>("RollingMean").constructor<int, std::string>();
class_<Macd,        base<EvalOp>>("Macd").constructor<int, int, int>();
```

Everything else is inherited. This is why the spike measured only ~340 to 460 bytes marginal per operator, a constructor plus a vtable. The per-operator code-generation burden collapses to one line whose only variable is the constructor signature.

**Batch correctness comes for free.** A uniform `eval`-loop over the pure contract produces results identical to Python's optimized batch paths, guaranteed by screamer's batch-equals-stream invariant. So v1 needs no per-operator batch kernels. The extracted N-to-M kernels from Phase 1 become an optional performance refinement. One nuance: the sliding-extremum block algorithm is a batch-only optimization, so the eval-loop gives the right numbers without the O(n) trick, which is acceptable for v1.

**Code generation: one manifest, three emitters.** The signature and default information the generators need lives in the binding `nb::init<>` declarations, not in any current introspection (the existing `generate_screamer__init__.py` extracts names only, `generate_screamer__init__.py:51-54`). Introduce a checked-in op manifest, `{name, header, cpp_class, ctor:[{name, cpp_type, ts_type, default}], n_in, n_out, category}`, derived from the uniform binding declarations by a libclang parse. From it, generate:

1. `bindings_wasm.cpp`, the constructor registrations above.
2. `screamer.d.ts`, the typed operator declarations.
3. the JS op table, `n_in`/`n_out` and parameter names/defaults the dispatch wrapper needs.

A CI freshness gate regenerates and diffs, so the JS layer cannot silently drift from Python. Because Phase 1 already rewires all 238 registrations for the split, the same manifest can optionally regenerate the nanobind bindings too, unifying both bindings on one source of truth. Treat that unification as a follow-on, not a v1 blocker.

**The DAG/Pipeline is the one non-uniform Embind surface:** a small hand-written set of classes (`Pipeline`, `Node`, `Input`, combinators, `Resample`) over the already-pure `dag/` engine.

## Layer 3: the JavaScript/TypeScript API

**Callable-factory pattern, for literal parity with Python.** Python has no `new`, so `screamer.RollingMean(3)` is a call. The JS binding mirrors that: a PascalCase factory returns a callable operator object, so call sites read the same in both languages.

```ts
import init, { RollingMean, Pipeline, Input } from "@screamer-labs/screamer";
await init();                                       // one-time async WASM load

using op = RollingMean(3);                           // callable and disposable
op(1.0);                                             // number        -> number
op(new Float64Array([1,2,3,4,5]));                   // Float64Array   -> Float64Array (batch, self-contained)
op([1,2,3]);                                         // number[]       -> number[]      (container-preserving)
for (const y of op(genOfNumbers())) { /* ... */ }    // Iterable       -> lazy generator
for await (const y of op(wsStream())) { /* ... */ }  // AsyncIterable  -> async generator
op.reset();
```

This preserves screamer's contract: the batch path resets internally and is independent (the `reset()` brackets in `functor_base.h`), while the scalar and iterable paths stream statefully. Returns are container-preserving (`Float64Array` to `Float64Array`, `number[]` to `number[]`). Multi-input operators take N arguments, the list-of-tuples form, and the 2-D `(T,N)` split, the same as Python.

**Multi-dimensional returns: the one genuine JS-vs-Python gap.** JS has no numpy, so the `(T,...,M)` arrays Python returns for multi-output or multi-column batch need a representation. v1 returns a `Float64Array` for the 1-D case (the overwhelming majority) and a zero-dependency `{ data: Float64Array, shape: number[] }` ndarray-lite for higher rank, with an optional `.toNested()` helper. This keeps the hot path allocation-free and parity-exact without a heavy dependency. It is the one place worth revisiting for ergonomics.

**Error model, cleaner than Python's.** Because the polymorphic dispatch lives in the TS layer, wrong-input-type errors throw directly in TS as `TypeError` with parity messages. Only genuine kernel errors (for example `RollingMean(0)`, a C++ `invalid_argument`) cross from C++, mapped to `RangeError`. Embind propagates C++ exceptions via `-fwasm-exceptions`; the TS layer normalizes them so callers never see raw Emscripten error objects.

**Object lifecycle: the Embind handle problem, solved three ways.** Each operator owns a WASM-heap C++ object, so the binding provides:

1. `using` and `Symbol.dispose` (TS 5.2) as the primary idiom, freeing at scope end.
2. explicit `.dispose()` for callers not using `using`.
3. a `FinalizationRegistry` best-effort net if a caller forgets.

Temporary batch buffers are always freed within the call, confirmed by the spike.

**Pipeline/DAG shape, define-then-bind, mirrored:**

```ts
const x = new Input();
const y = RollingMean(3)(x);                         // composes a Node, not a value
const p = new Pipeline(y);                            // plus combineLatest / resample nodes
p({ x: new Float64Array([/* ... */]) });             // bind data at call time
```

This maps one-to-one onto Python's `Input`/compose/bind engine over the pure `dag/` core.

## Binding technology: Embind, and why

Embind plus typed-array marshalling plus a TS ergonomics layer is the current state of the art for shipping a large C++ library to JavaScript. The closest analog is OpenCV.js: a large numeric C++ library exposed via auto-generated Embind with a TS layer, which validates both the Embind choice and the code-generation-from-a-manifest choice. Skia/CanvasKit, ammo.js, and Box2D use Embind; sql.js, TF.js, ONNX Runtime Web, and DuckDB-Wasm use typed-array heap marshalling with a TS layer on a thin core.

The one genuinely novel element is the uniform `EvalOp`-base runtime, a domain-specific simplification screamer's uniform contract allows, in the spirit of minimizing per-symbol Embind glue.

**The alternative worth recording: a raw C ABI** (`extern "C"` plus `EMSCRIPTEN_KEEPALIVE`) with hand or generated JS glue, the smaller and faster school that sql.js and FFmpeg.wasm follow. Embind carries a fixed runtime cost and a per-class registration cost, measured at ~340 to 460 bytes per operator and a ~17 KB baseline, which is acceptable at our scale. If binary size ever becomes critical, the C ABI is the escape hatch, at the cost of more hand-written glue. For 248 uniform operators with code generation and a measured-acceptable Embind cost, Embind is the right call. The WebAssembly Component Model with WIT and jco is the standards-track future, but it is still maturing for browser class-based C++ APIs and is a v2 re-platform consideration, not a first-release choice.

## Packaging and distribution

- **Module format:** ESM-first, a CJS build for older Node, dual browser and Node targets, shipped `.d.ts`, a `package.json` `exports` map with `import`/`require`/`types` conditions.
- **WASM loading:** default to a separate `.wasm` resolved via `import.meta.url`, plus a single-file inlined build (`-sSINGLE_FILE=1`) as an alternate export for bundler-hostile or zero-config setups. Bundler asset resolution is the top integration pain for WASM libraries, so both are offered.
- **Bundle strategy:** a monolithic `.wasm` with all operators. WASM does not tree-shake at the JS bundler boundary, so the binary loads whole regardless of which operators are imported. The JS/TS wrapper layer does tree-shake, so importing only `RollingMean` drops the other wrappers from the consumer's JS bundle while the full `.wasm` still loads. At the projected size this is the right default. Per-family `.wasm` splitting would re-pay the ~17 KB baseline per split and is a future option only.
- **Size budget, from the spike:** wasm ~55 to 90 KB gzipped, JS glue ~6 to 10 KB gzipped after `--closure=1` (which took the glue from 43.5 to 18 KB raw), plus the imported TS wrappers. About 65 to 100 KB gzipped all-in for the full library. Levers: `-Oz`, emmalloc, `--closure=1`, `-sFILESYSTEM=0`.
- **Build and release:** a separate CMake/CI target, not part of the Python wheel, producing the npm artifacts in a dedicated `wasm` CI job, published via npm with OIDC provenance, mirroring the PyPI trusted-publishing flow. The op manifest, codegen freshness gate, and parity suite gate that job.

## Testing and parity

The oracle is the Python package. The correctness bet is JS output equals Python output across the registry.

1. **Golden-vector parity (the core).** A devtools script runs the Python screamer to emit reference input/output vectors for every operator and every regime: scalar streaming, batch, lazy, multi-in/out, plus warmup, NaN, and reset behaviors, and edge inputs (all-NaN, empty, length-1, inf, extreme windows). These commit as goldens. The JS suite replays the same inputs and asserts equality. Both bindings compile the identical kernels, so results should be bit for bit; a tight ULP tolerance guards against any x86-vs-wasm difference, though both use strict IEEE-754 f64 with no fast-math, so determinism is expected.
2. **Invariants re-run JS-side:** causality, reset restores initial state, and batch-equals-stream equivalence, validating the JS dispatch layer, not only the kernel. This is the same registry-driven approach as `test_contract_compliance.py`, enumerated from the manifest.
3. **JS-specific tests:** `dispose`/`using` frees with no leak (checked via Emscripten heap counters), `FinalizationRegistry`, error mapping, container-preserving return types, async-generator over a mock async source, and Pipeline define-then-bind.
4. **CI matrix:** Node across several versions plus headless Chromium via Playwright, since browser WASM and typed-array behavior can differ. Parity and invariants gate the npm publish.
5. **Codegen freshness gate:** CI regenerates the Embind `.cpp`, `.d.ts`, and op table from the manifest and diffs, so a new Python operator cannot ship without its JS peer.

## Delivery phases

Phase 1 is a standalone Python release. Phases 2 through 6 are internal milestones toward the single all-in-one npm release, each independently reviewable and testable. Sequencing follows the dependency chain 1, 2, 3, then 4 and 5 partly in parallel, then 6.

**Phase 1: base-class split.** Extract the pure batch kernels out of `FunctorBase`'s `handle_input_*_numpy`; split `base.h`, `functor_base.h`, and `detail::` into pure-compute and nanobind headers; relocate dispatch; rewire the 238 registrations. Verified by the 6926 Python tests staying green and byte-identical, and by a standalone compile proving an operator header builds with no binding library. Ships as a normal screamer 2.x release. Improves the Python side and de-risks everything downstream.

**Phase 2: manifest and codegen.** The libclang-parsed op manifest, the three emitters, and the CI freshness gate. Verified by the manifest round-tripping the nanobind bindings and the generated Embind compiling.

**Phase 3: Embind runtime and emcc build.** The uniform `EvalOp`-base runtime plus generated per-operator constructors; the size-optimized emcc target with the single-file variant. Verified by the full `.wasm` building, a smoke test constructing and running one event for every operator, and a measured full-library size checked against the 55 to 90 KB projection.

**Phase 4: JS/TS API and lifecycle.** The callable-factory dispatch across all four regimes, container-preserving, the ndarray-lite returns, error mapping, `using`/dispose/FinalizationRegistry, async init, and multi-in/out. Verified by the JS-specific suite.

**Phase 5: DAG/Pipeline surface.** The hand-written Embind classes over the pure `dag/` engine and the define-then-bind JS API. Verified by Pipeline parity tests, a Python pipeline versus the JS pipeline on the same data.

**Phase 6: parity, CI, packaging, publish.** Golden generation from the Python oracle across the registry; the parity and invariant suites; Node and headless-browser CI; npm packaging and OIDC publish wired into the version-bump and tag flow. Verified by the full parity suite green on Node and Chromium. Ships the all-in-one `@screamer-labs/screamer` 2.x npm release.

## Risk

Risk concentrates in three places. Phase 1's N-to-M kernel extraction is a batch-parity risk, netted by the 6926 tests. Phase 3's binary size is a real number to measure, not assume. Phase 5's DAG surface is the only substantial hand-written binding code. Everything else is mechanical or code-generation owned.

## Open items to confirm during planning

- The multi-dimensional return representation (ndarray-lite versus nested arrays versus a typed-ndarray dependency) is an ergonomics call that can be revisited.
- Whether to unify the nanobind bindings onto the manifest in Phase 2 or leave that as a follow-on.
- The exact libclang extraction versus a hand-authored manifest seed for the first pass.
