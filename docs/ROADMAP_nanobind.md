# Roadmap: migrate bindings to nanobind

**Status**: parked. Not blocking. Revisit when wheel-matrix maintenance becomes
painful or when a Python release exposes the lack of forward compat.

## Why this is on the roadmap

The C++ binding layer (`bindings/*.cpp`, `include/screamer/common/*.h`) is
written against [pybind11](https://pybind11.readthedocs.io/) v2.13.6. Each new
Python minor release (3.15, 3.16, …) currently requires us to:

1. Add a new entry to `[tool.cibuildwheel] build = "..."` in `pyproject.toml`
2. Cut a new release tag so wheels get rebuilt and uploaded to `pypi.deep.fund`
3. Anyone using the new Python in the meantime falls back to source builds
   (slow, requires a C++17 toolchain)

Multiplied across 4 OSes, every Python release adds 4 wheels per release. The
matrix grows without bound.

The fix is to ship **abi3 / Py_LIMITED_API** wheels — built once against the
CPython Limited API, valid forever on 3.x ≥ minimum. **pybind11 cannot do this
reliably today.** nanobind can.

## What we tried (May 2026)

We attempted to enable the Python Limited API with pybind11 v2.13.6:

```cmake
pybind11_add_module(screamer_bindings MODULE STABLE_ABI ${SOURCES})  # broken
```

CMake error: `No SOURCES given to target: screamer_bindings`. `STABLE_ABI`
isn't in pybind11's `cmake_parse_arguments` options list (we checked
`build/_deps/pybind11-src/tools/pybind11Tools.cmake` — recognised options are
only `MODULE;SHARED;EXCLUDE_FROM_ALL;NO_EXTRAS;SYSTEM;THIN_LTO;OPT_SIZE`).
The same is true in pybind11 v3.0 (CMakeLists keyword set:
`STATIC;SHARED;MODULE;THIN_LTO;OPT_SIZE;NO_EXTRAS;WITHOUT_SOABI`).

The pybind11 maintainers track this in
[discussion #4474 — "Support for stable CPython ABI?"](https://github.com/pybind/pybind11/discussions/4474),
which has been open since 2023. The blocker is that pybind11's class-binding
machinery historically reached into private CPython structures (tuple/list
internals, dict slots) that the Limited API doesn't expose. Until that's
refactored, stable ABI is out of reach for pybind11.

Manual workarounds (define `Py_LIMITED_API` ourselves, tag wheels via
`scikit_build.wheel.py-api`) compile but produce extensions that crash at
runtime when pybind11 hits one of those private structures. Not viable for a
financial library that needs reliability.

## nanobind: the realistic alternative

[nanobind](https://nanobind.readthedocs.io/) is by the same author
(Wenzel Jakob), positioned as the "modern, faster, slimmer successor" to
pybind11. It has **first-class stable ABI support** with a documented minimum
of Python 3.12:

```cmake
nanobind_add_module(my_ext STABLE_ABI source1.cpp source2.cpp)
```

The result is a single `cp312-abi3-*.whl` per OS that works on 3.12, 3.13,
3.14, 3.15, … without recompilation. cibuildwheel detects abi3 and runs tests
against every Python in `CIBW_BUILD` using the same wheel.

### What we'd gain

| Today | After migration |
|---|---|
| 4 OS × 4 Python versions = **16 wheels** per release | 4 OS × 1 abi3 wheel = **4 wheels** per release (for 3.12+) |
| New Python (3.15+) needs new wheels | Existing abi3 wheel just works |
| ~3.5 GB CI artifacts per release | ~1 GB |
| Build time per release ≈ 25–45 min | ≈ 8–12 min |
| pybind11 internal ABI bumps (e.g. 2.x→3.0) force coordinated re-releases | Stable ABI insulates us |

There's also a smaller win: nanobind benchmarks ~2× faster argument parsing
than pybind11 and ~3× smaller binary sizes. For a streaming numerics library
where the boundary cost matters less than the inner C++ loop, that's nice but
not the headline reason.

### What it would cost

Roughly **a focused 1–2 day session**. The API surface we use is small and
nanobind mirrors pybind11 closely.

Files needing edits (line counts approximate):

| File | Lines | Migration character |
|---|---|---|
| `bindings/bindings.cpp` | 30 | Mechanical: `pybind11` → `nanobind`, `py::module` → `nb::module_` |
| `bindings/bindings_core.cpp` | 25 | Mechanical |
| `bindings/bindings_math.cpp` | 90 | Mechanical |
| `bindings/bindings_rolling.cpp` | 150 | Mechanical |
| `bindings/bindings_ew.cpp` | 130 | Mechanical |
| `bindings/bindings_preprocessing.cpp` | 35 | Mechanical |
| `bindings/bindings_signal.cpp` | 15 | Mechanical |
| `bindings/bindings_fin.cpp` | 30 | Mechanical |
| `bindings/bindings_misc.cpp` | 20 | Mechanical |
| `bindings/bindings_myfunctors.cpp` | 30 | Mechanical |
| `include/screamer/common/base.h` + `base.cpp` | 200 | **Real work** — `py::array_t<double>`, the polymorphic `operator()`, iterator/async-generator dispatch all need to be rewritten against nanobind's `nb::ndarray<>` API |
| `include/screamer/common/iterator.h` | 50 | Real work — nanobind's iterator protocol differs |
| `include/screamer/common/async_generator.h` + `.cpp` | 100 | Real work — async generator handling is the trickiest part |
| `include/screamer/common/functor_base.h` | 440 | Skip if MyFunctor is still parked; otherwise real work |
| `CMakeLists.txt` | 80 | Replace `pybind11` FetchContent with `nanobind`; switch `pybind11_add_module` → `nanobind_add_module(... STABLE_ABI ...)` |

### Known migration gotchas

1. **`py::array_t<double>` → `nb::ndarray<double>`** — nanobind's ndarray uses
   a tagged template (`nb::ndarray<double, nb::ndim<1>>` etc.) and stride
   handling is more explicit. Our `process_python_array` in `base.cpp` will
   need rewriting; the algorithm is fine but the surface API changes.
2. **`py::iterable`, `py::iterator`** — nanobind has these but the type names
   differ; iterator protocol is similar but not identical.
3. **Async generators** — nanobind's async support is less mature than
   pybind11's; we may need to keep some custom Python-side machinery.
4. **Stable ABI is opt-in but constrained**. Once we set `STABLE_ABI`, we
   can't use nanobind features that aren't covered by the limited API.
   The list is documented; for our use case (numerics, ndarray, iterators)
   everything we need is in scope.
5. **Minimum Python = 3.12** with `STABLE_ABI`. Anyone still on 3.11 would
   need either a separate non-abi3 wheel (defeats half the point) or to
   upgrade. By the time we migrate, 3.11 EOL (October 2027) may be close
   enough that dropping it is fine.

## When to revisit

Triggers in roughly order of likelihood:

1. **Python 3.15 ships (October 2026)** — if anyone wants to use it before we
   cut a release with `cp315-*` in the matrix, they'll hit source builds.
   Repeated complaints = migrate.
2. **CI minutes become a budget concern** — 16 wheels × 4–5 minutes each is
   reasonable today, but if the matrix grows or we add benchmarks to CI,
   abi3 cuts that to a quarter.
3. **pybind11 ships first-class STABLE_ABI** — if discussion #4474 ever
   becomes a feature, that's a much smaller migration than nanobind. Watch
   that thread.
4. **A library we depend on migrates first** — if we end up needing nanobind
   for some other reason, do the binding migration in the same session.

## Sketch of the migration plan when we do it

1. Branch `nanobind-migration` from main.
2. Add nanobind via FetchContent in `CMakeLists.txt`, keep pybind11 alongside
   for one commit so we can cross-reference the APIs.
3. Migrate `bindings/bindings.cpp` first (smallest, mostly mechanical) —
   prove the build pipeline.
4. Migrate `include/screamer/common/base.h` + `base.cpp` — the heart of the
   polymorphic dispatcher. This is the riskiest single file; do it second so
   if it doesn't pan out we haven't wasted work on the rest.
5. Migrate iterator + async generator code. Validate against
   `tests/test_stream_vs_generator.py`.
6. Bulk-migrate the remaining `bindings_*.cpp` files. Mostly mechanical.
7. Migrate `functor_base.h` if MyFunctor is back in scope by then; otherwise
   keep it pybind11 and excluded (it's already excluded today).
8. Drop pybind11 FetchContent. Set `requires-python = ">=3.12"`. Set
   `[tool.cibuildwheel] build = "cp312-*"`. Verify single abi3 wheel produced.
9. Test the abi3 wheel on 3.12, 3.13, 3.14 (cibuildwheel does this
   automatically with `test-command`).
10. Cut a major version bump (`make major` → `1.0.0`) to signal the API
    boundary change and the new Python floor.

Done. The wheel matrix shrinks 4× and stops growing.
