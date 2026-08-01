# nanobind Feasibility Spike Report

**Verdict: FEASIBLE. No hard blocker.** All 7 risky mechanisms compile and run
under nanobind. The spike built a standalone `nb_spike` module against the
pybind-free screamer headers and a Python smoke test exercises every path with
numerically-correct output.

- **nanobind version:** v2.9.2 (FetchContent git tag, `NB_VERSION 2.9.2`)
- **abi3 / STABLE_ABI:** BUILT. Produced `nb_spike.abi3.so`; imports and runs
  correctly on **both** CPython 3.12.12 and 3.13.8 from the *same* binary.
- **Toolchain:** AppleClang 17, C++17, CMake 3.x, arm64 macOS.
- **Clean build time (configure incl. nanobind git clone + full compile+link):**
  **~4.9 s** total wall (`cmake -S . -B build` + `cmake --build -j4`).
- **Env note:** the repo's default interpreter is Python **3.11.8**, on which
  abi3 is *impossible* (see item 6). The spike used a throwaway 3.12 venv with
  numpy to prove abi3. Everything else works on 3.11 too.

Spike location: `spike/` (`CMakeLists.txt`, `nb_spike.cpp`, `spike_helper.py`,
`smoke_test.py`). It does **not** touch the real `screamer_bindings` target.

---

## Per-item results

### 1. 1-in/1-out op through pure `EvalOp` + nanobind dispatch — WORKS

Wrapped the real `screamer::detail::RollingMean` kernel in a `MeanEvalOp : EvalOp`
(no Python). A `Mean` class does the type dispatch in its `__call__`, taking a
generic `nb::object` and branching (exactly mirroring `ScreamerBase`'s dispatch,
just relocated to the binding layer):

- (a) scalar `float`/`int` -> `float`: `nb::isinstance<nb::float_>(arg)` /
  `nb::isinstance<nb::int_>(arg)`, `nb::cast<double>(arg)`, return `nb::cast(out)`.
- (b) `list` -> `list`: `nb::isinstance<nb::list>`, iterate `nb::handle`, build
  `nb::list result; result.append(out);`.
- (c) 1-D ndarray -> **NEW** ndarray (the R1 owner-capsule crux, see below).
- (d) 2-D / Fortran-strided ndarray: read with per-axis element strides
  (`a.stride(ax)`), reset the op at each row boundary. Verified against a
  `np.asfortranarray` view (non-trivial strides) and it matches.
- (e) int64/int32 ndarray coerced to double: manual dtype switch (see gotcha #6).

`reset()` is provably called around array/list calls: a `reset_count` property
increments once per batch call and the smoke test asserts it.

**The R1 crux (returning a new numpy array).** There is no owning
`ndarray(shape)` constructor. You allocate, wrap in a capsule with a deleter, and
hand nanobind the pointer + shape + owner:

```cpp
double* outbuf = new double[total];
// ... fill outbuf (C-contiguous) ...
nb::capsule owner(outbuf, [](void* p) noexcept { delete[] (double*) p; });
return nb::cast(nb::ndarray<nb::numpy, double>(
    outbuf, /*ndim*/ shape.size(), /*shape*/ shape.data(), owner));
```

Note the extra `nb::cast(...)` — because the branch returns `nb::object`, the
`ndarray` must be explicitly cast to a Python object. (If the whole function's
return type were `nb::ndarray<...>` nanobind would auto-convert.) Reading the
input used the fully-generic `nb::ndarray<>` (any dtype/rank/strides) plus
`.ndim() / .shape(i) / .stride(i) / .data() / .dtype()`.

### 2. Multi-output op returning a tuple — WORKS

`MeanSumEvalOp : EvalOp` with `n_out()==2`. Scalar path returns
`std::make_tuple(out[0], out[1])` and nanobind converts it to a Python tuple via
`#include <nanobind/stl/tuple.h>`. Array path returns a NEW `(N, 2)` ndarray
(same capsule idiom, `shape{len, 2}`). Smoke test confirms the tuple type and the
`(5,2)` shape with a correct cumulative-sum second column.

### 3. `std::optional<double>` mutually-exclusive ctor + `std::vector<double>` ctor — WORKS

```cpp
nb::class_<Smoother>(m, "Smoother")
    .def(nb::init<std::optional<double>, std::optional<double>>(),
         "period"_a = nb::none(), "cutoff"_a = nb::none());
```

`Smoother(period=10.0)`, `Smoother(cutoff=0.25)`, and both-None / both-set (which
`raise ValueError`) all behave correctly. `None` flows in as an empty
`std::optional` via `#include <nanobind/stl/optional.h>`. The `Taps` class binds
`nb::init<const std::vector<double>&>()` via `#include <nanobind/stl/vector.h>`
and accepts a Python list.

### 4. Hand-written lazy iterator — WORKS

`LazyMeanIter` holds an `nb::object source` (the upstream Python iterator), its
own `EvalOp`, and a keep-alive handle. Bound as:

```cpp
nb::class_<LazyMeanIter>(m, "LazyMeanIter")
    .def("__iter__", &LazyMeanIter::iter, nb::rv_policy::reference_internal)
    .def("__next__", &LazyMeanIter::next);
```

`__next__` pulls from the source and translates StopIteration exactly as
required:

```cpp
try {
    item = source.attr("__next__")();
} catch (nb::python_error& e) {
    if (e.matches(PyExc_StopIteration)) throw;  // re-raise -> ends `for`
    throw;
}
```

`for x in Mean(3)(gen())` streams and produces the correct rolling values.

### 5. `nb::find(this)` to fetch the op's own Python wrapper — WORKS-WITH-DIFFERENT-API

pybind's `py::cast(this)` becomes **`nb::find(*this)`** — note it takes a
*reference*, not a pointer. Inside `Mean::call`, when the arg is a generic
iterator, we grab `nb::object self_wrapper = nb::find(*this);` and hand it to the
`LazyMeanIter` as a keep-alive so the parent op cannot be collected mid-stream.
The returned handle is the live wrapper (streaming through a transient parent op
works with no other reference held).

### 6. abi3 / STABLE_ABI — WORKS-WITH-DIFFERENT-API (important CMake gotcha)

`nanobind_add_module(nb_spike STABLE_ABI ...)` produced `nb_spike.abi3.so`, and
the single binary imports+runs on 3.12 **and** 3.13. **But it silently
downgrades** to a version-tagged `.cpython-312-darwin.so` unless CMake finds the
SABI component. nanobind's own condition:

```cmake
if ((Python_VERSION VERSION_LESS 3.12) OR (NOT TARGET Python::SABIModule))
    set(ARG_STABLE_ABI FALSE)   # no warning emitted
```

So the `find_package` line **must** request it:

```cmake
find_package(Python 3.8 COMPONENTS Interpreter Development.Module
             Development.SABIModule REQUIRED)
```

screamer's current root `CMakeLists.txt` uses
`find_package(Python3 COMPONENTS Interpreter Development.Module)` — migrating to
abi3 requires adding `Development.SABIModule` or abi3 quietly won't happen.

**`requires-python` implication: abi3 forces `>=3.12`.** On the repo's current
default 3.11 the STABLE_ABI request is dropped and you get a normal per-version
extension. So the migration has a policy choice: adopt abi3 (single wheel per
platform, but **drop Python 3.11** — pyproject currently declares `>=3.11`), or
keep building one wheel per minor version and forgo abi3.

### 7. Async-generator rework / replacing `py::exec` — WORKS (import mechanism proven)

`py::exec` / `py::eval` are gone in nanobind. The replacement is a real
importable Python module driven from C++:

```cpp
nb::module_ helper = nb::module_::import_("spike_helper");
return helper.attr("scale")(x);          // call a named function, no exec
```

`spike/spike_helper.py` holds `scale()` and an `async def stream(source, op)`
async-generator (the rework target). The C++ imports the module, calls the sync
function (verified: `call_helper(2.5) == 25.0`), and confirms the async-gen
function object is importable (`async_helper_importable() is True`). Full
end-to-end async driving was **not** wired up (not required by the spike), but
the load-bearing risk — that `py::exec` is removed — is fully retired: ship the
coroutine as a `.py` file and `import_` it. No blocker.

---

## `spike/CMakeLists.txt` approach

Minimal, self-contained, FetchContent-based; never references the real target.

```cmake
cmake_minimum_required(VERSION 3.18)
project(nb_spike LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Python 3.8 COMPONENTS Interpreter Development.Module
             Development.SABIModule REQUIRED)   # SABIModule is REQUIRED for abi3

include(FetchContent)
FetchContent_Declare(nanobind
  GIT_REPOSITORY https://github.com/wjakob/nanobind.git
  GIT_TAG v2.9.2)
FetchContent_MakeAvailable(nanobind)

nanobind_add_module(nb_spike STABLE_ABI
  nb_spike.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/../src/screamer/detail/start_policy.cpp)  # kernel dep
target_include_directories(nb_spike PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../include)
```

Configure/build:
```
cmake -S spike -B spike/build -DPython_EXECUTABLE=<py3.12>/bin/python
cmake --build spike/build -j4
PYTHONPATH=spike/build:spike <py3.12>/bin/python spike/smoke_test.py
```

---

## Top gotchas the migration plan must address

1. **abi3 needs `Development.SABIModule` in `find_package`, and forces
   `requires-python >=3.12`.** Without the component, STABLE_ABI silently
   downgrades (no warning). Adopting abi3 means dropping the currently-supported
   Python 3.11. This is a project policy decision, not a technical blocker.
2. **Returning ndarrays is manual.** No owning `ndarray(shape)` ctor: allocate,
   wrap in `nb::capsule` with a `delete[]` deleter, pass `data + ndim + shape +
   owner`, and `nb::cast(...)` it when the function returns `nb::object`. Every
   op that emits a numpy array (the whole batch path) needs this idiom. A shared
   helper (like `make_owned_array` here) is worth writing once.
3. **`nb::ndarray<>` does not auto-coerce dtype or require contiguity.** The int
   and Fortran-strided cases work only because the spike reads via an explicit
   `dtype.code/bits` switch and per-axis `stride(i)`. pybind's `py::array_t<double,
   forcecast>` used to hand you a contiguous double buffer for free; under
   nanobind you either write the strided/dtype reader yourself or rely on the
   typed-`ndarray` implicit-conversion pass. Decide one approach library-wide.

Minor: `nb::find` takes a reference (`nb::find(*this)`), not `py::cast(this)`;
`def_prop_ro` replaces `def_property_readonly`; all STL casters
(optional/vector/tuple/string) need explicit `#include`s or you get opaque
template errors. Dispatch on `nb::isinstance<nb::float_>` will *not* catch numpy
scalar types (e.g. `np.float64`) — the real dispatch must handle those as it does
today.

Report generated by the spike; binary + tests reproducible under `spike/`.
