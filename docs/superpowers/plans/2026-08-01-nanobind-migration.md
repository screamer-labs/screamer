# Nanobind Migration Plan (feasibility-validated)

**Status:** proposed. Feasibility proven by a compile spike on 2026-08-01 (all 7 risky mechanisms compiled and ran; abi3 built and imported on 3.12 and 3.13 from one binary). Supersedes the lost `docs/ROADMAP_nanobind.md` (2026-05, deleted in 34e5a0e).

**Spike artifacts:** `docs/superpowers/spikes/2026-08-01-nanobind/` (`REPORT.md`, `nb_spike.cpp`, `CMakeLists.txt`, `spike_helper.py`, `smoke_test.py`). The `nb_spike.cpp` is the reference implementation of every proven nanobind idiom.

## Why

pybind11 has no stable-ABI option (upstream issue open since 2023). nanobind has `STABLE_ABI` from Python 3.12+. Proven in the spike: one `nb_spike.abi3.so` imports and runs on CPython 3.12 and 3.13.

- **Release matrix collapse:** `build-wheels.yml` currently builds cp311-cp314 x {ubuntu, macos-14, windows} = ~12-16 wheels. abi3 collapses this to **one wheel per platform** (3 wheels) that covers 3.12+, built once.
- **Faster compile, smaller binary, lower per-call dispatch overhead** (nanobind's design goals). Serves the "efficiency is the product" positioning.
- **First step of the WASM-enabling refactor:** the migration forces the base-class split (Python dispatch out of `ScreamerBase`/`FunctorBase`, pure-compute base retained), which is the acknowledged prerequisite for a JS/WASM binding. Done once, both benefit.

## The one decision (needs the user): abi3 => drop Python 3.11

`STABLE_ABI` requires `requires-python >= 3.12`. screamer currently declares `>=3.11` and CI builds 3.11-3.14. The spike confirmed abi3 is silently dropped on 3.11.

- **Option A (recommended): adopt abi3, drop 3.11.** Full payoff (3 wheels, one build). 3.11 is in security-fix-only phase (EOL Oct 2027); 3.12 is broadly adopted. Ship the drop with a clear changelog note; a minor bump is defensible since dropping a Python version is not a public-API break, though a major (2.0) is the conservative signalling choice.
- **Option B: migrate to nanobind without abi3, keep 3.11.** Still gets faster compile / smaller binary / faster dispatch and the base-class split, but keeps per-version wheels and forgoes the matrix collapse (the biggest win).

This is the only policy fork; everything below assumes Option A and notes where B differs.

## Feasibility summary (spike, no hard blocker)

| Mechanism | Result | Idiom |
|---|---|---|
| 1-in/1-out dispatch over pure `EvalOp` | WORKS | dispatch relocated to binding layer |
| ndarray in -> NEW ndarray out | WORKS | allocate + `nb::capsule` deleter + `nb::cast` |
| strided / N-D / int-dtype input | WORKS | explicit `stride(i)` + dtype switch (no free `forcecast`) |
| multi-output tuple | WORKS | `nanobind/stl/tuple.h` |
| `std::optional` mutex ctor + `std::vector` ctor | WORKS | `nb::init<...>()`, `"a"_a = nb::none()`, stl includes |
| lazy `__iter__`/`__next__` + keep-alive | WORKS | `nb::rv_policy::reference_internal`, `nb::python_error`/`.matches` |
| `nb::find(*this)` (own wrapper) | WORKS (diff API) | takes a reference, not `py::cast(this)` |
| abi3 / STABLE_ABI | WORKS (CMake gotcha) | `find_package(... Development.SABIModule)` or it silently downgrades |
| async-gen `py::exec` replacement | WORKS | ship coroutine as `.py`, `nb::module_::import_(...)` it |

No multiple inheritance, holders, custom casters, trampolines, pickle, metaclasses, buffer protocol, or GIL scoping are used, so none of nanobind's removed features block the migration.

## Architecture: the base-class split

Today every op inherits `ScreamerBase`/`FunctorBase`, which `#include <pybind11>` and implement the scalar/list/ndarray/iterator dispatch inline. The migration relocates that dispatch into the binding layer:

- **Pure-compute base** (no Python): `EvalOp` is already pure (`n_in/n_out/eval(const double*, double*)/reset`). `ScreamerBase`'s `process_scalar`/`process_array_no_stride`/`process_array_stride` are already pure `double`-in/`double`-out. Strip the pybind members (`operator()`, `process_python_array`, `handle_input_*`) out of these headers so an op header compiles with no CPython present.
- **Binding-layer dispatcher** (nanobind): a single reusable dispatcher that wraps any `EvalOp` and provides `__call__` over scalar / list / ndarray / iterable / async-gen / dag-node, plus the lazy iterator classes. The spike's `nb_spike.cpp` `Mean`/`LazyMeanIter` are the template.

This split is the WASM prerequisite; it lands here as a side effect.

## Phases

### Phase 0 - decision + scaffolding
- User decides Option A vs B (drop 3.11 for abi3).
- Add nanobind via `FetchContent` (tag v2.9.2 as spiked; check for a newer stable tag at execution). Add `Development.SABIModule` to the `find_package(Python3 ...)` line. Keep pybind11 in the build in parallel during migration so the suite stays runnable throughout.
- Restore the `docs/ROADMAP_nanobind.md` reference dangling in `pyproject.toml` (point it at this plan).

### Phase 1 - the shared nanobind dispatch layer (the real work)
Build the nanobind equivalents of the 8 pybind-coupled headers as one dispatch module. Write these shared helpers once (spiked idioms):
- `make_owned_array(double* buf, shape...)` -> allocate + `nb::capsule([](void*p){delete[](double*)p;})` + typed `nb::ndarray<nb::numpy, double>` + `nb::cast`. Used by every batch path.
- An input reader that handles arbitrary dtype (int32/int64 -> double) and strides via `dtype().code/bits` + per-axis `stride(i)`. **Decision:** a hand-written strided/dtype reader (full `forcecast` parity) vs relying on typed-`nb::ndarray<double, c_contig>` implicit conversion (simpler, but copies and rejects some layouts). Recommend the strided reader for exact behavioral parity with today.
- The polymorphic dispatch, matching `ScreamerBase::operator()` and `FunctorBase::handle_input_*` behavior exactly, including: numpy scalar types (`np.float64`) which `nb::isinstance<nb::float_>` does NOT catch (keep the `can_cast_to_double` numpy-scalar check), list/tuple, 0-d array -> scalar, 2-D `(T,N)` split, and the error types the tests assert.
- Lazy iterator (`LazyEvalIterator`) + async iterator (`LazyAsyncIterator`/`AnextAwaitable`) with `nb::object` keep-alive via `nb::find(*this)`.
- **Async rework:** move the `process_awaitable` coroutine out of the `py::exec` string into a real `screamer/_async.py` helper, imported via `nb::module_::import_("screamer._async")`. This retires the one removed-feature dependency.

Validate this layer against 2-3 representative ops (a `ScreamerBase` 1-1, a `FunctorBase` multi-out, an optional-arg op) before the bulk.

### Phase 2 - base-class split
Split `common/base.h` and `common/functor_base.h` into (a) a pure-compute base header (included by all 174 op headers, no pybind) and (b) the nanobind dispatch wrapper (Phase 1). Confirm an op header now compiles under a no-Python toolchain (a quick standalone compile check, the WASM readiness signal).

### Phase 3 - mechanical bulk migration
Convert the 12 `bindings/*.cpp` (238 registrations, 3 uniform templates):
- `py::`->`nb::`, `PYBIND11_MODULE`->`NB_MODULE`, `py::module&`->`nb::module_&`, `py::init<>`->`nb::init<>`, `def_property_readonly`->`def_prop_ro`.
- Add explicit `#include <nanobind/stl/{optional,vector,string,tuple}.h>` where used.
- `py::arg("x") = py::none()` -> `"x"_a = nb::none()`; normalize the `std::nullopt` vs `py::none()` default inconsistency the inventory found (bindings_signal vs bindings_ew) to one style.
- `bindings_dag.cpp` lambda factory `py::init([]...)` -> nanobind placement-new `.def("__init__", [](T* t, ...){ new (t) T(...); })`; `py::array::c_style|forcecast` args -> typed `nb::ndarray` + `make_owned_array` outputs.

### Phase 4 - build / CI cutover
- Switch to `nanobind_add_module(screamer_bindings STABLE_ABI ...)`, drop the pybind11 fetch, set `requires-python >= 3.12` (Option A).
- `build-wheels.yml`: collapse the wheel matrix to one abi3 wheel per OS. KEEP a multi-Python **test** matrix (install the abi3 wheel and run `pytest` on 3.12 / 3.13 / 3.14) so the single wheel is verified across versions.
- `screamer/__init__.py` is autogenerated by introspecting the built module (downstream of the bindings) - it should regenerate unchanged; verify.

### Phase 5 - validation (the safety net)
The existing 6,831-test suite is the migration oracle. In particular the registry-wide invariant harness (`test_contract_compliance.py`: causality / reset / all-3-regimes; `test_nan_input_compliance.py`: nan-transparency / sticky / at-index) plus every per-op parity test must stay GREEN. Any behavioral drift surfaces there. Add:
- a test asserting the built wheel is abi3-tagged, and an import smoke test on 3.12/3.13/3.14 (CI already has the matrix).
- Watch the known deltas from the spike: numpy-scalar dispatch (`np.float64`), int-array coercion and non-contiguous strides, the exact exception types the dispatch raises (`py::cast_error`/`py::type_error` -> nanobind equivalents; `test_*` may assert on error type/message), and list-in -> list-out vs array-in -> array-out return-type parity.

## Effort (grounded)

The old "1-2 day" estimate assumed find/replace and predates the base-class-split realization; it is optimistic. Realistic, de-risked by the spike and the test safety net:
- Phase 1-2 (dispatch layer + base split) is the thinking: ~2-4 focused days.
- Phase 3 (mechanical bulk, 238 uniform registrations): ~1-2 days.
- Phase 4-5 (CI cutover + chasing parity deltas the suite flags): ~1-2 days.
- **~1 focused week.** This is execution risk, not feasibility risk: the spike proved every mechanism compiles and runs.

## Risks

- **Behavioral parity on the array path** (dtype coercion / strides) is the main risk; the invariant + parity tests are the net. Do Phase 1 with a couple of ops and run the relevant test files before the bulk.
- **Error-type/message parity** in the polymorphic dispatch (some tests assert `ValueError`/`TypeError`); map nanobind's exception vocabulary deliberately.
- **The drop-3.11 decision** is reversible only by forgoing abi3 (Option B); decide before Phase 4.
- Keeping pybind11 and nanobind both building during Phases 1-3 avoids a long red-suite window.
