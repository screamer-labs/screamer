# screamer.js Phase 1: Base-Class Split - Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make every screamer operator header compile with no binding library present, by splitting `ScreamerBase`, `FunctorBase`, and their `detail::` helpers into a pure-compute base (no nanobind) plus a relocated nanobind dispatch layer, while Python behavior stays byte-identical.

**Architecture:** The nanobind dispatch currently lives inside `common/base.h` and `common/functor_base.h`, which every operator header includes, so no operator header compiles without CPython. This plan makes those two headers pure and moves the dispatch (`operator()`, `handle_input*`, `process_python_array`, and the nanobind `detail::` helpers) into a new `common/dispatch.h` + `common/dispatch.cpp`. The Python batch loops move wholesale to the nanobind side and keep serving Python unchanged; nothing is extracted into shared kernels in this phase.

**Tech Stack:** C++17, nanobind, CMake, scikit-build-core, pytest (the 6926-test suite is the oracle).

## Global Constraints

- Python behavior must stay **byte-identical**. The oracle is the full suite: `make install-dev && pytest -q` must stay green (currently 6926 pass, 0 fail).
- The pure headers (`common/base.h`, `common/functor_base.h`, `common/eval_op.h`) and every operator header must compile with **no binding library on the include path**. This is the WASM-readiness gate and is a hard requirement.
- Do **not** change any operator's numeric behavior, dispatch semantics, error types, or public API. This is a pure relocation refactor.
- Do **not** extract the batch loops into shared pure kernels in this phase. They relocate to the nanobind side as-is. Extraction is a Phase 3 concern.
- Never hand-edit version files; only `make patch/minor/major` moves versions. This phase ships later as a normal screamer 2.x release; the version bump is out of scope for these tasks.
- Follow the existing include and naming conventions. Run `make tidy` before considering C++ work done (clang-tidy member-init gate).
- No em-dashes in any prose or comments (ASCII hyphens).

## File Structure

**Made pure (nanobind removed):**
- `include/screamer/common/base.h` - keeps only the `ScreamerBase` pure-compute contract (`process_scalar`, `process_array_no_stride`, `process_array_stride`, `eval`, `reset`, `n_in`, `n_out`). Removes `operator()`, `process_python_array`, all `detail::` nanobind helpers, and the nanobind/`cast_double`/`async_generator` includes.
- `include/screamer/common/functor_base.h` - keeps only the `FunctorBase` pure-compute contract (`call`, `process_columns`, `eval`, `reset`, `n_in`, `n_out`) plus the pure `detail::TupleOfDoubles`/`write_tuple_to_memory` helpers. Removes every `handle_input*` method, the nanobind `detail::` helpers (`read_n_arrays`, `maybe_split_TxN`), and the nanobind/`lazy_eval_iterator` includes. Adds `static constexpr size_t kN = N, kM = M;` so the relocated dispatch can read arity.
- `src/screamer/common/base.cpp` - keeps only the pure definitions `ScreamerBase::process_array_no_stride` and `ScreamerBase::process_array_stride`. Loses its nanobind includes.

**New (nanobind dispatch, moved out of the pure headers):**
- `include/screamer/common/dispatch.h` - the nanobind `detail::` helpers (`half_bits_to_float`, `load_elem`, `make_owned_array`, `is_ndarray`, `is_unsupported_dtype_array`, `coerce_to_f64`, `coerce_if_unsupported_dtype_array`, `coerce_args_if_unsupported`, `is_c_contiguous`, `ContigDouble`, `read_contig_double`, `read_n_arrays`, `maybe_split_TxN`), the `is_dag_node`/`make_dag_functor_node` declarations, the free function `screamer_call(ScreamerBase&, nb::object)`, the free function `process_python_array(ScreamerBase&, nb::ndarray<>)`, and the free function templates `functor_call<Op>(Op&, nb::args)` plus the relocated `handle_input_*` logic as free templates.
- `src/screamer/common/dispatch.cpp` - the definitions of `screamer_call`, `process_python_array`, `is_dag_node`, `make_dag_functor_node` (moved verbatim from `base.cpp`).

**Rewired:**
- `bindings/*.cpp` (12 files) - each `.def("__call__", &screamer::<Op>::operator())` becomes `.def("__call__", &screamer::screamer_call, ...)`; each `.def("__call__", &screamer::<Op>::handle_input)` becomes `.def("__call__", &screamer::functor_call<screamer::<Op>>)`. Add `#include "screamer/common/dispatch.h"`.
- `CMakeLists.txt` - add `src/screamer/common/dispatch.cpp` to the bindings sources; add a `check-pure-headers` custom target.

**New tooling:**
- `devtools/check_pure_headers.sh` - compiles a representative `ScreamerBase` op header and a `FunctorBase` op header with `-I include` and no nanobind, asserting they build. The WASM-readiness gate.
- `tests/test_pure_headers.py` - a pytest wrapper that runs the shell gate so CI enforces it.

---

### Task 1: Split the `detail::` helpers and `ScreamerBase` out of `base.h`

**Files:**
- Modify: `include/screamer/common/base.h` (strip to the pure `ScreamerBase` contract)
- Create: `include/screamer/common/dispatch.h` (nanobind helpers + `screamer_call` + `process_python_array` decls)
- Modify: `src/screamer/common/base.cpp` (keep only the two pure `process_array_*` definitions)
- Create: `src/screamer/common/dispatch.cpp` (`screamer_call`, `process_python_array`, `is_dag_node`, `make_dag_functor_node`)
- Modify: all `bindings/*.cpp` that register `ScreamerBase` ops (rewire `__call__`; add the dispatch include)
- Modify: `CMakeLists.txt` (add `dispatch.cpp`)

**Interfaces:**
- Produces:
  - `screamer::ScreamerBase` (pure): `virtual double process_scalar(double) = 0;`, `virtual void process_array_no_stride(double*, const double*, size_t);`, `virtual void process_array_stride(double*, size_t, const double*, size_t, size_t);`, plus `eval`/`reset`/`n_in`/`n_out` from `EvalOp`. No nanobind.
  - `nb::object screamer::screamer_call(ScreamerBase& self, nb::object obj);` - the relocated `operator()` body, unchanged logic. Bound as `__call__`.
  - `nb::object screamer::process_python_array(ScreamerBase& self, nb::ndarray<> input);` - the relocated batch entry.
  - The nanobind `detail::` helpers, moved verbatim into `dispatch.h` under `namespace screamer::detail`.

- [ ] **Step 1: Baseline the suite is green before touching anything**

Run: `make install-dev && pytest -q 2>&1 | tail -5`
Expected: all pass (about 6926 passed). Record the exact pass count.

- [ ] **Step 2: Create `include/screamer/common/dispatch.h` with the nanobind helpers and dispatch decls**

Move the entire `namespace detail { ... }` block currently in `base.h` (lines 25-216: `half_bits_to_float`, `load_elem`, `make_owned_array`, `is_ndarray`, `is_unsupported_dtype_array`, `coerce_to_f64`, `coerce_if_unsupported_dtype_array`, `coerce_args_if_unsupported`, `is_c_contiguous`, `ContigDouble`, `read_contig_double`) verbatim into `dispatch.h`. Also move the `is_dag_node` / `make_dag_functor_node` declarations. Add the new free-function declarations. Skeleton:

```cpp
#ifndef SCREAMER_DISPATCH_H
#define SCREAMER_DISPATCH_H

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>
#include <sstream>
#include "screamer/common/base.h"          // pure ScreamerBase
#include "screamer/common/cast_double.h"
#include "screamer/common/async_generator.h"

namespace nb = nanobind;

namespace screamer {

bool is_dag_node(const nb::object& obj);
nb::object make_dag_functor_node(nb::object self, nb::object args_tuple);

namespace detail {
    // ... moved verbatim from base.h lines 25-216 ...
}  // namespace detail

// Relocated ScreamerBase dispatch (was ScreamerBase::operator() and
// ScreamerBase::process_python_array). Free functions taking the base by
// reference so one function pointer binds every ScreamerBase op as __call__.
nb::object screamer_call(ScreamerBase& self, nb::object obj);
nb::object process_python_array(ScreamerBase& self, nb::ndarray<> input);

}  // namespace screamer
#endif
```

- [ ] **Step 3: Strip `base.h` to the pure contract**

Rewrite `include/screamer/common/base.h` to remove the nanobind include, the `cast_double`/`async_generator` includes, the `namespace detail` block, the `is_dag_node`/`make_dag_functor_node` decls, and the `operator()`/`process_python_array` members. Final content:

```cpp
#ifndef SCREAMER_BASE_H
#define SCREAMER_BASE_H

#include <cstddef>
#include "screamer/common/eval_op.h"

namespace screamer {

class ScreamerBase : public EvalOp {
public:
    virtual ~ScreamerBase() = default;

    void reset() override {}
    std::size_t n_in() const override { return 1; }
    std::size_t n_out() const override { return 1; }
    void eval(const double* in, double* out) override { out[0] = process_scalar(in[0]); }

    virtual double process_scalar(double value) = 0;

    virtual void process_array_no_stride(double* result_data, const double* input_data, size_t size);
    virtual void process_array_stride(
        double* result_data, size_t result_stride,
        const double* input_data, size_t input_stride, size_t size);
};

}  // namespace screamer
#endif
```

- [ ] **Step 4: Split `base.cpp` into a pure `base.cpp` and a nanobind `dispatch.cpp`**

In `src/screamer/common/base.cpp`, keep only the two pure definitions and drop all nanobind includes:

```cpp
#include "screamer/common/base.h"
#include <cstddef>

namespace screamer {

void ScreamerBase::process_array_no_stride(double* result_data, const double* input_data, size_t size) {
    for (size_t i = 0; i < size; i++) result_data[i] = process_scalar(input_data[i]);
}

void ScreamerBase::process_array_stride(
    double* result_data, size_t result_stride,
    const double* input_data, size_t input_stride, size_t size) {
    size_t r = 0, in = 0;
    for (size_t i = 0; i < size; i++) { result_data[r] = process_scalar(input_data[in]); r += result_stride; in += input_stride; }
}

}  // namespace screamer
```

Create `src/screamer/common/dispatch.cpp` holding, moved verbatim from the old `base.cpp`: `is_dag_node`, `make_dag_functor_node`, the body of the old `ScreamerBase::operator()` renamed to `nb::object screamer_call(ScreamerBase& self, nb::object obj)` (replace `process_scalar` with `self.process_scalar`, `reset()` with `self.reset()`, `process_python_array(arr)` with `process_python_array(self, arr)`, `nb::find(*this)` with `nb::find(self)`), and the body of `ScreamerBase::process_python_array` renamed to `nb::object process_python_array(ScreamerBase& self, nb::ndarray<> input)` (replace `reset()`/`process_array_no_stride`/`process_array_stride` with `self.`-qualified calls). Includes:

```cpp
#include "screamer/common/dispatch.h"
#include "screamer/common/lazy_eval_iterator.h"
#include <nanobind/stl/vector.h>
#include <stdexcept>
#include <sstream>
#include <vector>
```

- [ ] **Step 5: Rewire `ScreamerBase`-op `__call__` bindings and add the dispatch include**

In each `bindings/*.cpp`, add `#include "screamer/common/dispatch.h"` near the other includes, then replace every `&screamer::<AnyType>::operator()` with `&screamer::screamer_call`. Because `screamer_call` takes `ScreamerBase&`, one pointer binds every ScreamerBase op (including the templated `Transform<...>` ops). Command to find them:

Run: `grep -rn '::operator()' bindings/`
Then apply, per file: `perl -0pi -e 's/&screamer::[^,()]*::operator\(\)/&screamer::screamer_call/g' bindings/bindings_*.cpp`
Expected: every ScreamerBase `__call__` now points at `screamer_call`. FunctorBase `handle_input` bindings are untouched (Task 2).

- [ ] **Step 6: Add `dispatch.cpp` to the build**

In `CMakeLists.txt`, find the list of binding/common source files (where `src/screamer/common/base.cpp` appears) and add `src/screamer/common/dispatch.cpp` next to it.

Run: `grep -n 'common/base.cpp' CMakeLists.txt`
Then add the sibling line for `dispatch.cpp`.

- [ ] **Step 7: Build and run the full suite; confirm byte-identical behavior**

Run: `make install-dev && pytest -q 2>&1 | tail -5`
Expected: identical pass count to Step 1 (about 6926 passed, 0 failed). Any failure here is a relocation bug (a mis-qualified `self.` call or a missing include); fix before proceeding.

- [ ] **Step 8: Run clang-tidy**

Run: `make tidy`
Expected: no errors.

- [ ] **Step 9: Commit**

```bash
git add include/screamer/common/base.h include/screamer/common/dispatch.h \
        src/screamer/common/base.cpp src/screamer/common/dispatch.cpp \
        bindings/ CMakeLists.txt
git commit -m "refactor(base): split ScreamerBase into pure base + nanobind dispatch"
```

---

### Task 2: Split `FunctorBase` out of `functor_base.h`

**Files:**
- Modify: `include/screamer/common/functor_base.h` (strip to the pure `FunctorBase` contract)
- Modify: `include/screamer/common/dispatch.h` (add the relocated `functor_call<Op>` + `handle_input_*` free templates + `read_n_arrays`/`maybe_split_TxN`)
- Modify: all `bindings/*.cpp` that register `FunctorBase` ops (rewire `__call__`)

**Interfaces:**
- Consumes: `screamer::ScreamerBase` and the `detail::` helpers from Task 1's `dispatch.h`.
- Produces:
  - `screamer::FunctorBase<Derived, N, M>` (pure): `virtual ResultTuple call(const InputArray&) = 0;`, `virtual bool process_columns(...)`, `eval`/`reset`/`n_in`/`n_out`, the pure `detail::TupleOfDoubles`/`write_tuple_to_memory`, and `static constexpr size_t kN = N; static constexpr size_t kM = M;`. No nanobind.
  - `template <class Op> nb::object screamer::functor_call(Op& self, nb::args args);` - the relocated `FunctorBase::handle_input`, dispatching on `Op::kN`/`Op::kM`. Bound as `__call__` for every FunctorBase op.

- [ ] **Step 1: Confirm the suite is green (post-Task-1 baseline)**

Run: `pytest -q 2>&1 | tail -3`
Expected: about 6926 passed.

- [ ] **Step 2: Move the nanobind `detail::` functor helpers into `dispatch.h`**

Cut `read_n_arrays<N>` and `maybe_split_TxN<N>` (functor_base.h:62-118) from `functor_base.h` and paste them into the `namespace detail` block of `dispatch.h`. Leave `TupleOfDoublesHelper`, `TupleOfDoubles`, `write_tuple_helper`, `write_tuple_to_memory` (functor_base.h:29-56) in `functor_base.h` (they are pure and used by `eval`).

- [ ] **Step 3: Strip `functor_base.h` to the pure contract**

Remove from `functor_base.h`: the nanobind and `lazy_eval_iterator` includes; every `handle_input_*` method (the `_numpy` variants at lines 172-300 and the dispatch variants at 306-651); the private nanobind helpers `is_series_array`, `is_lazy_iterable`, `eager_parallel`, `cast_to_array`, `args_to_tuple_n`. Keep `call`, `process_columns`, `eval`, `reset`, `n_in`, `n_out`, the `using` type aliases, and the pure `detail` tuple helpers. Add the arity constants. The class becomes:

```cpp
template <class Derived, size_t N, size_t M>
class FunctorBase : public EvalOp {
public:
    static constexpr size_t kN = N;
    static constexpr size_t kM = M;
    using InputArray  = std::array<double, N>;
    using OutputArray = std::array<double, M>;
    using ResultTuple = std::conditional_t<M == 1, double, typename detail::TupleOfDoubles<M>>;

    virtual bool process_columns(
        double*, std::ptrdiff_t, const std::array<double*, N>&,
        const std::array<int64_t, N>&, const std::array<size_t, N>&, size_t, size_t) { return false; }

    virtual ResultTuple call(const InputArray& inputs) = 0;
    void reset() override {}

    std::size_t n_in() const override { return N; }
    std::size_t n_out() const override { return M; }
    void eval(const double* in, double* out) override {
        InputArray inputs;
        for (std::size_t i = 0; i < N; ++i) inputs[i] = in[i];
        if constexpr (M == 1) out[0] = call(inputs);
        else detail::write_tuple_to_memory(out, call(inputs));
    }
};
```

Includes reduce to: `<array> <cstddef> <optional> <stdexcept> <string> <tuple> <type_traits> <utility> <vector> <cstdint>` plus `"screamer/common/eval_op.h"`. No nanobind, no `base.h`.

- [ ] **Step 4: Relocate the `handle_input*` machinery into `dispatch.h` as free templates**

In `dispatch.h`, add a `functor_call<Op>` free template plus the moved `handle_input_*` logic as free templates parameterized on `Op` (reading `Op::kN`/`Op::kM`). Mechanical transform of each moved method: the method body is copied verbatim, then `this->call(...)` / `call(...)` becomes `self.call(...)`, `reset()` becomes `self.reset()`, `process_columns(...)` becomes `self.process_columns(...)`, `static_cast<Derived*>(this)` becomes `&self`, `nb::find(*static_cast<Derived*>(this))` becomes `nb::find(self)`, and `InputArray`/`ResultTuple` become `typename Op::InputArray`/`typename Op::ResultTuple`. The private helpers `is_series_array`, `is_lazy_iterable`, `eager_parallel<Op>`, `cast_to_array<Op>`, `args_to_tuple_n` move as free functions/templates too. The entry point:

```cpp
template <class Op>
nb::object functor_call(Op& self, nb::args args) {
    constexpr size_t N = Op::kN, M = Op::kM;
    for (nb::handle a : args) {
        if (screamer::is_dag_node(nb::borrow<nb::object>(a)))
            return screamer::make_dag_functor_node(nb::find(self), nb::cast<nb::tuple>(args));
    }
    if constexpr (N == 1) { if (args.size() != 1) throw nb::type_error("Wrong number of in puts"); }
    if constexpr (N == 1 && M == 1) return functor_1i_1o<Op>(self, nb::borrow<nb::object>(args[0]));
    else if constexpr (N > 1 && M == 1) return functor_Ni_1o<Op>(self, args);
    else if constexpr (N == 1 && M > 1) return functor_1i_Mo<Op>(self, nb::borrow<nb::object>(args[0]));
    else return functor_Ni_Mo<Op>(self, args);
}
```

Preserve the exact error strings (including the existing "Wrong number of in puts" typo) and the exact branch order, so behavior is identical. Include `"screamer/common/lazy_eval_iterator.h"` in `dispatch.h` for the `LazyEvalIterator` used by the lazy branches.

- [ ] **Step 5: Rewire `FunctorBase`-op `__call__` bindings**

Replace every `&screamer::<Op>::handle_input` with `&screamer::functor_call<screamer::<Op>>`:

Run: `grep -rn '::handle_input' bindings/`
Then: `perl -0pi -e 's/&screamer::(\w+)::handle_input/&screamer::functor_call<screamer::$1>/g' bindings/bindings_*.cpp`
Expected: every FunctorBase `__call__` now points at `functor_call<Op>`.

- [ ] **Step 6: Build and run the full suite; confirm byte-identical behavior**

Run: `make install-dev && pytest -q 2>&1 | tail -5`
Expected: identical pass count to Task 1 Step 7 (about 6926 passed, 0 failed). Watch specifically the multi-input and multi-output op tests and the lazy/async iterator tests, which exercise the relocated `handle_input_*` paths.

- [ ] **Step 7: clang-tidy, then commit**

Run: `make tidy`
Expected: no errors.

```bash
git add include/screamer/common/functor_base.h include/screamer/common/dispatch.h bindings/
git commit -m "refactor(functor): split FunctorBase into pure base + relocated nanobind dispatch"
```

---

### Task 3: Add the no-binding compile gate (WASM-readiness)

**Files:**
- Create: `devtools/check_pure_headers.sh`
- Create: `tests/test_pure_headers.py`
- Modify: `CMakeLists.txt` (a `check-pure-headers` custom target)

**Interfaces:**
- Consumes: the pure `common/base.h` and `common/functor_base.h` from Tasks 1-2.
- Produces: a CI-enforceable gate proving an operator header compiles with `-I include` and no nanobind.

- [ ] **Step 1: Write the failing gate**

Create `devtools/check_pure_headers.sh`:

```bash
#!/usr/bin/env bash
# Prove operator headers compile with NO binding library present. This is the
# WASM-readiness invariant: op headers depend only on the pure compute base.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CXX="${CXX:-c++}"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

cat > "$tmp/pure.cpp" <<'EOF'
// A ScreamerBase op and a FunctorBase op, compiled with no nanobind on the path.
#include "screamer/rolling_mean.h"       // ScreamerBase, overrides process_array_no_stride
#include "screamer/rolling_min_max.h"    // FunctorBase<_,1,2>
int main() {
    screamer::RollingMean a(3);
    screamer::RollingMinMax b(3);
    double in = 1.0, out2[2];
    a.eval(&in, out2);
    b.eval(&in, out2);
    a.reset(); b.reset();
    return 0;
}
EOF

"$CXX" -std=c++17 -I "$ROOT/include" -c "$tmp/pure.cpp" -o "$tmp/pure.o"
echo "OK: operator headers compile with no binding library."
```

Make it executable: `chmod +x devtools/check_pure_headers.sh`

- [ ] **Step 2: Run it against the split headers**

Run: `./devtools/check_pure_headers.sh`
Expected: `OK: operator headers compile with no binding library.` (If it fails with a nanobind include error, a pure header still leaks a binding include; fix it in `base.h`/`functor_base.h`.)

- [ ] **Step 3: Wrap it as a pytest so CI enforces it**

Create `tests/test_pure_headers.py`:

```python
import subprocess, pathlib
def test_operator_headers_compile_without_binding_library():
    root = pathlib.Path(__file__).resolve().parent.parent
    script = root / "devtools" / "check_pure_headers.sh"
    r = subprocess.run(["bash", str(script)], capture_output=True, text=True)
    assert r.returncode == 0, f"pure-header gate failed:\n{r.stdout}\n{r.stderr}"
```

- [ ] **Step 4: Run the gate test**

Run: `pytest tests/test_pure_headers.py -q`
Expected: 1 passed.

- [ ] **Step 5: Add a CMake convenience target**

In `CMakeLists.txt`, add:

```cmake
add_custom_target(check-pure-headers
    COMMAND ${CMAKE_COMMAND} -E env CXX=${CMAKE_CXX_COMPILER}
            bash ${CMAKE_SOURCE_DIR}/devtools/check_pure_headers.sh
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    COMMENT "Compile operator headers with no binding library (WASM-readiness gate)")
```

- [ ] **Step 6: Commit**

```bash
git add devtools/check_pure_headers.sh tests/test_pure_headers.py CMakeLists.txt
git commit -m "test: gate operator headers compile with no binding library (WASM-readiness)"
```

---

### Task 4: Final validation and changelog

**Files:**
- Modify: `CHANGELOG.md` (or the equivalent unreleased-changes note)

**Interfaces:**
- Consumes: all of Tasks 1-3.

- [ ] **Step 1: Full clean build and suite**

Run: `make clean && make install-dev && pytest -q 2>&1 | tail -5`
Expected: about 6926 passed, 0 failed, matching the Task 1 Step 1 baseline exactly.

- [ ] **Step 2: Run the WASM-readiness gate once more from clean**

Run: `./devtools/check_pure_headers.sh`
Expected: `OK: operator headers compile with no binding library.`

- [ ] **Step 3: clang-tidy**

Run: `make tidy`
Expected: no errors.

- [ ] **Step 4: Add a changelog entry (no version bump)**

Add an entry under the unreleased section of `CHANGELOG.md` describing the internal refactor: the base-class split, that Python behavior is unchanged, and that operator headers now compile with no binding library as groundwork for the JS/WASM binding. Do not edit any version string.

- [ ] **Step 5: Commit**

```bash
git add CHANGELOG.md
git commit -m "docs(changelog): base-class split, operator headers now binding-free"
```

---

## Notes for the executor

- Tasks 1 and 2 are the substance; each is verified by the full suite staying byte-identical to the Step-1 baseline. The relocation is mechanical, so any test failure points at a specific mis-transformed call (`self.`-qualification, a moved-but-not-included helper, or a changed error string).
- The single most likely failure mode is a pure header still transitively including nanobind. The Task 3 gate is what catches it; run it early and often, not only at the end.
- Do not "improve" the relocated dispatch (no dedup, no renamed errors, no reordered branches). This phase is behavior-preserving relocation only. Improvements belong to later phases.
- If the `perl` rewrites miss a templated op (for example a `Transform<...>` whose `operator()` spans characters the regex excludes), fix those by hand; the compiler will name them.
