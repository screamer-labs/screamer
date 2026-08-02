# screamer.js Phase 2: Codegen'd Embind Layer + emcc Build - Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a real `screamer.wasm` from all ~230 point operators via a code-generated Embind layer on the uniform `EvalOp` base, with every operator constructible and single-event-evaluable from Node, smoke-tested against the Python oracle, and the full-library binary size measured.

**Architecture:** An op manifest (parsed by a devtools script from the existing `bindings/*.cpp`) drives a generator that emits `bindings_wasm.cpp`: one `emscripten::class_<Op, base<EvalOp>>("Name").constructor<...>()` per operator. The JS-facing runtime (`evalInto`, `evalBatchInto`, `reset`, `nIn`, `nOut`, heap helpers) is bound ONCE on `EvalOp`, inherited by every op. A dedicated CMake configuration compiled with the Emscripten toolchain links the generated bindings against the pure C++ sources into `screamer.wasm` + an ES module.

**Tech Stack:** C++17, Emscripten (emcc 4.0.21) + Embind, CMake (Emscripten.cmake toolchain), Python (devtools codegen), Node (smoke test), the existing Python screamer package (parity oracle).

## Global Constraints

- This is JS-side / build-only work. It MUST NOT modify any Python binding, operator header, or C++ compute code, and must not change the Python suite (6927 passed, 2 skipped stays exactly).
- The Embind layer is generated, not hand-maintained. The only hand-written C++ is the uniform `EvalOp` runtime and the small explicit handlers for edge-case constructors.
- Scope is the point operators only. EXCLUDE `bindings_streams.cpp` and `bindings_dag.cpp` (the Pipeline/DAG surface is Phase 5); their `nb::list` constructors are out of scope here.
- The WASM build compiles ONLY pure C++ sources plus the generated bindings. It must NOT compile `src/screamer/common/dispatch.cpp` or `src/screamer/common/async_generator.cpp` (both nanobind) and must not link nanobind or CPython.
- Emscripten toolchain: `~/Projects/emsdk/upstream/emscripten/cmake/Modules/Platform/Emscripten.cmake`; activate with `source ~/Projects/emsdk/emsdk_env.sh` or call `emcmake`/`emcc` directly (both on PATH already; `emcc --version` = 4.0.21).
- No em-dashes in prose or comments (ASCII hyphens). Do not edit any version file.

## Op inventory (grounded)

- ~230 point ops across `bindings/{rolling,ew,math,fin,micro,signal,expanding,misc,preprocessing,supervised,backtest}.cpp`. Each registration is `nb::class_<CppType, screamer::ScreamerBase>` (1-in/1-out) or `nb::class_<CppType>` deriving `FunctorBase` (N-in/M-out), followed by `.def(nb::init<...>(), ...)` and `.def("__call__", &screamer::{screamer_call|functor_call<Op>})`.
- Constructor shapes: 84 `nb::init<>` (+ ~80 more default-constructed), 44 `<int>`, 25 `<int, const std::string&>`, and a long tail (`<int,double>`, `<double,double,double,double>`, etc.).
- Three edge cases the manifest must annotate:
  1. **26 templated `Transform<...>` math ops** in `bindings_math.cpp` (e.g. `nb::class_<screamer::Transform<(double(*)(double))std::abs>, screamer::ScreamerBase>(m, "Abs")`). Zero-arg ctor; the exact C++ type string must be captured verbatim.
  2. **~10 EW ops** in `bindings_ew.cpp` with `std::optional<double>` decay-param ctors. Embind cannot take `std::optional` directly; these get a custom Embind constructor (below).
  3. Ops with no explicit `nb::init<>` (default-constructed) -> `.constructor<>()`.

## File Structure

- Create `devtools/wasm/gen_wasm_manifest.py` - parses `bindings/*.cpp` (excluding streams/dag) into `devtools/wasm/wasm_manifest.json`.
- Create `devtools/wasm/wasm_manifest.json` - the committed manifest (regenerated + diffed in CI).
- Create `devtools/wasm/gen_wasm_bindings.py` - emits `wasm/generated/bindings_wasm.cpp` from the manifest.
- Create `wasm/embind_runtime.h` - the hand-written uniform `EvalOp` runtime + heap helpers + the custom EW-optional constructors.
- Create `wasm/generated/bindings_wasm.cpp` - generated (committed so the build is reproducible without re-running codegen).
- Create `wasm/CMakeLists.txt` - the emcc build config (pure sources + generated bindings -> `screamer.wasm` + `screamer.mjs`).
- Create `wasm/build-wasm.sh` - wraps `emcmake cmake` + build; prints the `.wasm` size.
- Create `wasm/smoke/smoke.mjs` - Node smoke test: construct + one-event-eval every op; compare a sample to Python.
- Create `devtools/wasm/gen_oracle.py` - emits `wasm/smoke/oracle.json` (a sample of ops with input -> expected single-event output from the Python package).
- Create `tests/test_wasm_codegen_fresh.py` - CI freshness gate: regenerate manifest + bindings, assert no diff.

---

### Task 1: Op manifest generator

**Files:**
- Create: `devtools/wasm/gen_wasm_manifest.py`
- Create: `devtools/wasm/wasm_manifest.json` (generated output, committed)

**Interfaces:**
- Produces: `wasm_manifest.json` = a list of `{ "name": str, "cpp_type": str, "base": "ScreamerBase"|"FunctorBase", "ctor": [str], "ctor_kind": "plain"|"transform"|"ew_optional", "source": str }`. `ctor` holds the C++ arg types from `nb::init<...>` with `const ...&` normalized to the bare type (e.g. `const std::string&` -> `std::string`); `ctor` is `[]` for `nb::init<>` / default-constructed. `cpp_type` is the exact type inside `nb::class_<HERE, ...>`.

- [ ] **Step 1: Write the parser**

`gen_wasm_manifest.py`: read every `bindings/*.cpp` except `bindings_streams.cpp` and `bindings_dag.cpp`. For each `nb::class_<TYPE, screamer::ScreamerBase>(m, "NAME")` or `nb::class_<TYPE>(m, "NAME")` (where TYPE derives FunctorBase - detect by absence of `screamer::ScreamerBase` and presence of a following `functor_call<` or `handle_input`-family `__call__`), capture NAME and TYPE. Then find the associated `.def(nb::init<ARGS>()...)` (may span multiple lines; join lines between `nb::init<` and the matching `>`), split ARGS at top-level commas (respecting `<>` nesting and `(...)` in function-pointer types), normalize `const X&` -> `X`. Classify `ctor_kind`: `transform` if TYPE starts with `screamer::Transform<`; `ew_optional` if any arg is `std::optional<double>`; else `plain`. Emit sorted-by-name JSON.

- [ ] **Step 2: Generate and eyeball**

Run: `python3 devtools/wasm/gen_wasm_manifest.py`
Expected: writes `devtools/wasm/wasm_manifest.json`.

- [ ] **Step 3: Write a validation test inline in the generator's `__main__`**

Assert: manifest length is >= 200; the `RollingMean` entry equals `{"name":"RollingMean","cpp_type":"screamer::RollingMean","base":"ScreamerBase","ctor":["int","std::string"],"ctor_kind":"plain",...}`; the `Abs` entry has `ctor_kind":"transform"`, `ctor":[]`, and `cpp_type` starting `screamer::Transform<`; at least one `ew_optional` entry exists (e.g. `EwMean`). Print `MANIFEST OK: <n> ops`.

- [ ] **Step 4: Run the validation**

Run: `python3 devtools/wasm/gen_wasm_manifest.py --check`
Expected: `MANIFEST OK: <n> ops` with n around 230.

- [ ] **Step 5: Commit**

```bash
git add devtools/wasm/gen_wasm_manifest.py devtools/wasm/wasm_manifest.json
git commit -m "feat(wasm): op manifest generator parsed from the nanobind bindings"
```

---

### Task 2: Uniform EvalOp Embind runtime

**Files:**
- Create: `wasm/embind_runtime.h`

**Interfaces:**
- Produces: an Embind registration block for `EvalOp` bound once, plus free functions. Consumed by the generated `bindings_wasm.cpp` (Task 3) via `#include "embind_runtime.h"` and a macro/inline `register_eval_op_runtime()`.
- Key signatures: `void evalInto(screamer::EvalOp& op, uintptr_t inPtr, uintptr_t outPtr)`; `void evalBatchInto(screamer::EvalOp& op, uintptr_t inPtr, uintptr_t outPtr, size_t rows)`; `uintptr_t allocF64(size_t n)`; `void freeBuf(uintptr_t)`; `emscripten::val viewF64(uintptr_t, size_t)`; plus custom EW constructors `screamer::EwMean* make_EwMean(int, double, double, double, double)` etc. using a NaN sentinel (NaN arg -> `std::nullopt`).

- [ ] **Step 1: Write the runtime header**

`embind_runtime.h` (uses the spike's proven idioms verbatim - `typed_memory_view`, `uintptr_t` heap pointers):

```cpp
#pragma once
#include <emscripten/bind.h>
#include <emscripten/val.h>
#include <cstddef>
#include <cstdint>
#include <cmath>
#include <optional>
#include "screamer/common/eval_op.h"

namespace screamer_wasm {

// One event: read n_in doubles at inPtr, write n_out doubles at outPtr.
inline void evalInto(screamer::EvalOp& op, std::uintptr_t inPtr, std::uintptr_t outPtr) {
    op.eval(reinterpret_cast<const double*>(inPtr), reinterpret_cast<double*>(outPtr));
}

// Batch by the batch==stream invariant: replay eval() over `rows` events.
// Layout: inPtr = rows * n_in() doubles (row-major), outPtr = rows * n_out().
inline void evalBatchInto(screamer::EvalOp& op, std::uintptr_t inPtr,
                          std::uintptr_t outPtr, std::size_t rows) {
    const double* in = reinterpret_cast<const double*>(inPtr);
    double* out = reinterpret_cast<double*>(outPtr);
    const std::size_t ni = op.n_in(), no = op.n_out();
    op.reset();
    for (std::size_t r = 0; r < rows; ++r) op.eval(in + r * ni, out + r * no);
}

inline std::uintptr_t allocF64(std::size_t n) {
    return reinterpret_cast<std::uintptr_t>(std::malloc(n * sizeof(double)));
}
inline void freeBuf(std::uintptr_t p) { std::free(reinterpret_cast<void*>(p)); }
inline emscripten::val viewF64(std::uintptr_t p, std::size_t n) {
    return emscripten::val(emscripten::typed_memory_view(n, reinterpret_cast<double*>(p)));
}

// A NaN sentinel maps "argument not provided" to std::nullopt for the EW ops
// whose Python ctors take std::optional<double> decay parameters.
inline std::optional<double> opt(double v) {
    return std::isnan(v) ? std::nullopt : std::optional<double>(v);
}

}  // namespace screamer_wasm

// Register the shared runtime on the EvalOp base class ONCE. Every op class is
// registered with base<EvalOp>, so all ops inherit evalInto/evalBatchInto/etc.
#define SCREAMER_REGISTER_EVAL_OP_RUNTIME()                                    \
    emscripten::class_<screamer::EvalOp>("EvalOp")                             \
        .function("evalInto", &screamer_wasm::evalInto, emscripten::allow_raw_pointers()) \
        .function("evalBatchInto", &screamer_wasm::evalBatchInto, emscripten::allow_raw_pointers()) \
        .function("reset", &screamer::EvalOp::reset)                           \
        .function("nIn", &screamer::EvalOp::n_in)                              \
        .function("nOut", &screamer::EvalOp::n_out);                           \
    emscripten::function("allocF64", &screamer_wasm::allocF64);               \
    emscripten::function("freeBuf", &screamer_wasm::freeBuf);                 \
    emscripten::function("viewF64", &screamer_wasm::viewF64)
```

- [ ] **Step 2: Standalone compile check under emcc**

Write a throwaway `wasm/_rt_check.cpp` = `#include "embind_runtime.h"\nEMSCRIPTEN_BINDINGS(x){ SCREAMER_REGISTER_EVAL_OP_RUNTIME(); }` and compile:

Run: `emcc wasm/_rt_check.cpp -std=c++17 -I include -lembind -sMODULARIZE=1 -sEXPORT_ES6=1 -sENVIRONMENT=node -o /tmp/rt.mjs && echo RT_OK`
Expected: `RT_OK`. Then `rm wasm/_rt_check.cpp /tmp/rt.*`.

- [ ] **Step 3: Commit**

```bash
git add wasm/embind_runtime.h
git commit -m "feat(wasm): uniform EvalOp Embind runtime (evalInto/evalBatchInto/reset/heap)"
```

---

### Task 3: Embind bindings generator

**Files:**
- Create: `devtools/wasm/gen_wasm_bindings.py`
- Create: `wasm/generated/bindings_wasm.cpp` (generated, committed)

**Interfaces:**
- Consumes: `wasm_manifest.json` (Task 1), `embind_runtime.h` (Task 2, its `opt()` and the runtime macro).
- Produces: `bindings_wasm.cpp` with one `emscripten::class_<CppType, emscripten::base<screamer::EvalOp>>("Name").constructor<...>()` per op, and custom `.constructor(&make_X, allow_raw_pointers())` for `ew_optional` ops.

- [ ] **Step 1: Write the generator**

`gen_wasm_bindings.py`: for each manifest entry emit, inside one `EMSCRIPTEN_BINDINGS(screamer)` block that starts with `SCREAMER_REGISTER_EVAL_OP_RUNTIME();`:
- `plain`/`transform`: `emscripten::class_<CppType, emscripten::base<screamer::EvalOp>>("Name").constructor<CTOR_TYPES>();` (empty `<>` for no-arg).
- `ew_optional`: emit a free factory in an anonymous namespace above the bindings, `inline CppType* make_Name(<doubles for each optional, plain types otherwise>) { return new CppType(<opt(x) for optional args, x otherwise>); }`, then `emscripten::class_<CppType, emscripten::base<screamer::EvalOp>>("Name").constructor(&make_Name, emscripten::allow_raw_pointers());`. (The int leading arg for EW ops stays `int`; only the `std::optional<double>` slots take the NaN sentinel.)
Include `"embind_runtime.h"` and every operator header the ops need (derive the header set: each op's header is `screamer/<snake_case-or-known>.h`; simplest and robust is to `#include "screamer/screamer.h"`-style umbrella if one exists, else include the union of headers already imported by the nanobind `bindings/*.cpp` - the generator can copy the `#include "screamer/..."` lines from those binding files, minus base/dispatch/functor_base). Prefer copying the op-header include lines from the source bindings so the set is exact.

- [ ] **Step 2: Generate**

Run: `python3 devtools/wasm/gen_wasm_bindings.py`
Expected: writes `wasm/generated/bindings_wasm.cpp`.

- [ ] **Step 3: Validate coverage inline**

In the generator `__main__`, assert every manifest op name appears exactly once as a `class_<...>("Name")` in the output, and that each `ew_optional` op has a matching `make_<Name>` factory. Print `BINDINGS OK: <n> classes, <k> custom ctors`.

- [ ] **Step 4: Run coverage check**

Run: `python3 devtools/wasm/gen_wasm_bindings.py --check`
Expected: `BINDINGS OK: <n> classes, <k> custom ctors` with n around 230, k around 10.

- [ ] **Step 5: Commit**

```bash
git add devtools/wasm/gen_wasm_bindings.py wasm/generated/bindings_wasm.cpp
git commit -m "feat(wasm): Embind bindings generator + generated bindings_wasm.cpp"
```

---

### Task 4: emcc CMake build

**Files:**
- Create: `wasm/CMakeLists.txt`
- Create: `wasm/build-wasm.sh`

**Interfaces:**
- Consumes: `wasm/generated/bindings_wasm.cpp`, `wasm/embind_runtime.h`, the pure C++ sources.
- Produces: `wasm/build/screamer.mjs` + `wasm/build/screamer.wasm`.

- [ ] **Step 1: Write `wasm/CMakeLists.txt`**

A standalone CMake project (not the Python one). Collect the PURE sources: `src/screamer/common/base.cpp` and everything under `src/screamer/detail/` and `src/screamer/dag/` that compiles without nanobind, EXPLICITLY EXCLUDING `src/screamer/common/dispatch.cpp` and `src/screamer/common/async_generator.cpp`. Use an explicit `set(PURE_SOURCES ...)` list (globbing then removing the two nanobind files is acceptable). Target: `add_executable(screamer wasm/generated/bindings_wasm.cpp ${PURE_SOURCES})` with `target_include_directories(screamer PRIVATE ${CMAKE_SOURCE_DIR}/include)`, `target_compile_features(... cxx_std_17)`, and link/link-options:

```cmake
set(EMCC_LINK "-lembind -sMODULARIZE=1 -sEXPORT_ES6=1 -sENVIRONMENT=node \
-sFILESYSTEM=0 -sALLOW_MEMORY_GROWTH=1 -sMALLOC=emmalloc -sEXPORTED_RUNTIME_METHODS=['HEAPF64']")
set_target_properties(screamer PROPERTIES SUFFIX ".mjs" LINK_FLAGS "${EMCC_LINK} -Oz")
target_compile_options(screamer PRIVATE -Oz)
```

- [ ] **Step 2: Write `wasm/build-wasm.sh`**

```bash
#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD="$ROOT/wasm/build"
rm -rf "$BUILD"; mkdir -p "$BUILD"
emcmake cmake -S "$ROOT/wasm" -B "$BUILD" -DCMAKE_SOURCE_DIR="$ROOT" >/dev/null
cmake --build "$BUILD" -j
sz() { stat -f%z "$1" 2>/dev/null || stat -c%s "$1"; }
gz() { gzip -9 -c "$1" | wc -c | tr -d ' '; }
echo "screamer.wasm raw=$(sz "$BUILD/screamer.wasm") gzip=$(gz "$BUILD/screamer.wasm")"
```
(Pass the repo root to CMake so it can resolve `include/` and `src/`; adjust `CMakeLists.txt` to read `CMAKE_SOURCE_DIR` or an explicit `-DSCREAMER_ROOT`.)

- [ ] **Step 3: Build**

Run: `bash wasm/build-wasm.sh`
Expected: compiles all ops, links, prints `screamer.wasm raw=<bytes> gzip=<bytes>`. If a specific op header fails to compile under emcc, that is a real finding - report which op and why (it would indicate a non-pure dependency the Phase 1 gate did not cover). If linking fails on a missing symbol, add the corresponding pure `.cpp` to `PURE_SOURCES`.

- [ ] **Step 4: Record the size**

Note the printed raw + gzip sizes; they are a Phase 2 deliverable (compare to the spec's 55-90 KB gzip projection).

- [ ] **Step 5: Commit**

```bash
git add wasm/CMakeLists.txt wasm/build-wasm.sh
git commit -m "feat(wasm): emcc CMake build -> screamer.wasm from all point ops"
```

---

### Task 5: Node smoke test against the Python oracle

**Files:**
- Create: `devtools/wasm/gen_oracle.py`
- Create: `wasm/smoke/oracle.json` (generated, committed)
- Create: `wasm/smoke/smoke.mjs`

**Interfaces:**
- Consumes: `wasm/build/screamer.mjs`, the manifest, the Python screamer package.
- Produces: a passing Node smoke run: every op constructs + evaluates one event without error; a sampled subset matches Python within 1e-9.

- [ ] **Step 1: Oracle generator**

`gen_oracle.py`: import the Python `screamer`; for a curated sample of ~20 ops spanning ScreamerBase and FunctorBase and both plain/transform/ew_optional kinds (e.g. RollingMean(3), RollingStd(3), Abs(), EwMean(span=5), MACD(...), RollingMinMax(3)), feed a fixed input series and record the single-event outputs after warmup (drive the op event-by-event, capture output at a chosen index). Emit `oracle.json`: `[{ "name","args","inputs":[[...]],"expect":[...] }]` where `inputs` is per-event n_in-tuples and `expect` is the per-event n_out output at the checked index. Use `poetry run python devtools/wasm/gen_oracle.py`.

- [ ] **Step 2: Smoke test**

`smoke.mjs`: `import init from "../build/screamer.mjs"; const M = await init();`. Part A (coverage): for every manifest op, construct it (use its default ctor if `ctor` is empty; otherwise a canonical arg vector: ints->3, doubles->1.0, strings->"strict", ew NaN-sentinel->NaN for optionals) via `new M[name](...)`, then run one event through `evalInto` using `allocF64`/`viewF64`/`freeBuf` with `nIn()`/`nOut()` sized buffers; assert no throw and a finite-or-NaN result. Part B (parity): for each `oracle.json` entry, drive the events and assert the checked output equals `expect` within 1e-9. Print `SMOKE OK: <n> ops constructed+evaluated, <k> parity checks matched`.

- [ ] **Step 3: Run it**

Run: `bash wasm/build-wasm.sh && node wasm/smoke/smoke.mjs`
Expected: `SMOKE OK: ~230 ops constructed+evaluated, ~20 parity checks matched`. A construct/eval failure on a specific op is a real finding (report the op + error). A parity mismatch is a real finding (report op + got/expected).

- [ ] **Step 4: Commit**

```bash
git add devtools/wasm/gen_oracle.py wasm/smoke/oracle.json wasm/smoke/smoke.mjs
git commit -m "test(wasm): Node smoke test - all ops construct+eval, sample matches Python"
```

---

### Task 6: Codegen freshness gate + Python-suite guard

**Files:**
- Create: `tests/test_wasm_codegen_fresh.py`

**Interfaces:**
- Consumes: the two generators (Tasks 1, 3).

- [ ] **Step 1: Freshness gate**

`tests/test_wasm_codegen_fresh.py`: run `gen_wasm_manifest.py` and `gen_wasm_bindings.py` to temp outputs and assert byte-identical to the committed `wasm_manifest.json` and `wasm/generated/bindings_wasm.cpp`. This makes a stale generated layer fail CI (so adding a Python op without regenerating is caught). Skip the emcc build here (CI builds WASM in a dedicated job in Phase 6); this test only guards the generated text.

- [ ] **Step 2: Run it**

Run: `poetry run pytest tests/test_wasm_codegen_fresh.py -q`
Expected: 1 passed (or 2 if split per generator).

- [ ] **Step 3: Confirm the Python suite is untouched**

Run: `poetry run pytest -q 2>&1 | tail -3`
Expected: 6928 passed, 2 skipped (the prior 6927 + this 1 new freshness test), matching that no Python behavior changed.

- [ ] **Step 4: Commit**

```bash
git add tests/test_wasm_codegen_fresh.py
git commit -m "test(wasm): codegen freshness gate (manifest + bindings regenerate clean)"
```

---

## Notes for the executor

- The risk concentrates in Task 3 (the `ew_optional` custom ctors and getting the exact op-header include set) and Task 4 (an op header that fails under emcc would reveal a non-pure dependency the Phase 1 gate missed - report it, do not weaken the build). The `Transform<...>` ops are easy (exact type string from the manifest, zero-arg ctor).
- Do not hand-edit `wasm/generated/bindings_wasm.cpp` or `wasm_manifest.json`; always regenerate. The freshness gate enforces this.
- The batch path is a uniform `evalBatchInto` eval-loop; correctness follows from screamer's batch==stream invariant, so the smoke test's per-event checks are sufficient here. Optimized per-op batch kernels are a later, optional refinement.
- Everything here is additive under `wasm/` and `devtools/wasm/` plus two new `tests/` files. No Python binding, operator header, or compute code changes.
