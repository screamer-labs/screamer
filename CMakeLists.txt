cmake_minimum_required(VERSION 3.18)
project(screamer VERSION 3.9.1 LANGUAGES CXX)

# Set the default build type to Release for optimizations
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()


# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# macOS deployment target floor: 10.13 (High Sierra, September 2017).
# We use std::optional::value(), which Apple's libc++ marks as unavailable
# when targeting < 10.13 (it depends on bad_optional_access exception
# machinery shipped with libc++abi.dylib starting in 10.13). cibuildwheel
# defaults to 10.9 for cp311-macosx_x86_64 wheels, which trips this. Force
# a 10.13 floor here so the build is robust regardless of how the
# deployment target gets set externally. Only bumps UP, never down — arm64
# already requires >= 11.0 and that's preserved.
if(APPLE)
    if(NOT CMAKE_OSX_DEPLOYMENT_TARGET OR CMAKE_OSX_DEPLOYMENT_TARGET VERSION_LESS "10.13")
        set(CMAKE_OSX_DEPLOYMENT_TARGET "10.13" CACHE STRING "macOS deployment target" FORCE)
    endif()
endif()

# Build with CPU-specific instructions (-march=native). OFF by default so
# wheels are portable; ON for local dev builds where you want max perf.
option(SCREAMER_NATIVE_ARCH "Build with -march=native (not portable)" OFF)


# Find Python and Development libraries.
# Development.SABIModule is REQUIRED for nanobind's STABLE_ABI (abi3). Without
# it, nanobind silently downgrades STABLE_ABI to a version-tagged extension
# (no warning). abi3 also forces requires-python >= 3.12 (see pyproject.toml).
find_package(Python 3.12 COMPONENTS Interpreter Development.Module Development.SABIModule REQUIRED)

# Include FetchContent module
include(FetchContent)

# Define a directory for dependencies to avoid redundant downloads
set(FETCHCONTENT_BASE_DIR ${CMAKE_BINARY_DIR}/_deps)

# Path to locally cached dependencies (must be populated manually or during an online run)
set(LOCAL_CACHE_DIR ${CMAKE_SOURCE_DIR}/_local_deps)

# nanobind replaces pybind11 (migration 2026-08). nanobind has STABLE_ABI from
# Python 3.12+, which collapses the release wheel matrix to one abi3 wheel per
# platform. v3.1.0 (bumped from v2.13.0) brings faster attribute lookups and
# function dispatch. See docs/superpowers/plans/2026-08-01-nanobind-migration.md.
#
# Pinned by commit, with a local patch, cmake/nanobind-surrogate-module.patch,
# for crashes of v3.1.0 that are fixed in open upstream PRs (remove each hunk
# when its fix is released):
#  1. wjakob/nanobind#1455: an error message holding a str that UTF-8 cannot
#     encode (a lone surrogate such as '\udc80', from a class's __module__ or an
#     exception message) segfaulted. Buffer::put_pystr() escapes it instead.
#     The patch also uses it in nb_type_init() (nb_type.cpp), where subclassing
#     a bound type with such a __module__ segfaulted; not yet reported upstream.
#  2. wjakob/nanobind#1454: make_mt_from_buffer_protocol() (nb_ndarray.cpp) read
#     view->strides[i] without a NULL check. A ctypes array and a NumPy
#     datetime64/timedelta64 scalar export a buffer with strides == NULL, so any
#     nb::ndarray argument given one segfaulted. The patch reads NULL strides as
#     C-contiguous. screamer also refuses those values before any ndarray cast
#     (TypeError, cast_double.h is_foreign_value).
# A version bump must re-check that the patch still applies (configure fails
# loudly when it does not).
set(SCREAMER_NANOBIND_VERSION "3.1.0")
set(SCREAMER_NANOBIND_PATCH "${CMAKE_SOURCE_DIR}/cmake/nanobind-surrogate-module.patch")

if(EXISTS "${LOCAL_CACHE_DIR}/nanobind")
    # A local copy must be the pinned version, and gets the same patch.
    set(_nb_dir "${LOCAL_CACHE_DIR}/nanobind")
    file(STRINGS "${_nb_dir}/include/nanobind/nanobind.h" _nb_ver
         REGEX "#define NB_VERSION_(MAJOR|MINOR|PATCH) ")
    string(REGEX REPLACE ".*MAJOR ([0-9]+).*MINOR ([0-9]+).*PATCH ([0-9]+).*"
           "\\1.\\2.\\3" _nb_ver "${_nb_ver}")
    if(NOT _nb_ver VERSION_EQUAL SCREAMER_NANOBIND_VERSION)
        message(FATAL_ERROR "${_nb_dir} is nanobind ${_nb_ver}; screamer needs "
                            "${SCREAMER_NANOBIND_VERSION}. Replace or delete it.")
    endif()
    execute_process(
        COMMAND "${CMAKE_COMMAND}" -DSRC=${_nb_dir} -DPATCH=${SCREAMER_NANOBIND_PATCH}
                -P "${CMAKE_SOURCE_DIR}/cmake/apply_nanobind_patch.cmake"
        RESULT_VARIABLE _nb_patch)
    if(NOT _nb_patch EQUAL 0)
        message(FATAL_ERROR "Could not apply ${SCREAMER_NANOBIND_PATCH} to ${_nb_dir}")
    endif()
    message(STATUS "Using locally cached nanobind ${_nb_ver} at ${_nb_dir}")
    add_subdirectory("${_nb_dir}" "${CMAKE_BINARY_DIR}/nanobind")
else()
    message(STATUS "Downloading nanobind ${SCREAMER_NANOBIND_VERSION} from GitHub")
    FetchContent_Declare(
        nanobind
        GIT_REPOSITORY      https://github.com/wjakob/nanobind.git
        GIT_TAG             68480a9e6883bb1bf65e1925423321e2a89d07f9  # v3.1.0
        UPDATE_DISCONNECTED ON
        PATCH_COMMAND       "${CMAKE_COMMAND}" -DSRC=<SOURCE_DIR>
                            -DPATCH=${SCREAMER_NANOBIND_PATCH}
                            -P "${CMAKE_SOURCE_DIR}/cmake/apply_nanobind_patch.cmake"
    )
    FetchContent_MakeAvailable(nanobind)
endif()


# ---------------------------------------------------------
# Create the Python module
# ---------------------------------------------------------

# All binding translation units (converted to nanobind, migration 2026-08).
file(GLOB_RECURSE BINDING_SOURCES "bindings/*.cpp")

# Collect all source files in the src directory
file(GLOB_RECURSE SCREAMER_SOURCES "src/screamer/*.cpp")


nanobind_add_module(screamer_bindings STABLE_ABI ${BINDING_SOURCES} ${SCREAMER_SOURCES})
target_include_directories(
    screamer_bindings PUBLIC
    "${CMAKE_SOURCE_DIR}/include"
)

# Optimization flags. MSVC takes different flags than GCC/Clang.
if(MSVC)
    # Plain /O2. Skip /fp:fast (analogue of GCC -ffast-math, which we
    # established breaks our NaN-sentinel pattern). Skip whole-program
    # optimisation (/GL via INTERPROCEDURAL_OPTIMIZATION) to rule it out
    # as the cause of the cp311+ "PyEval_SaveThread: GIL not held"
    # crash on import. Can be re-enabled later once Windows is green.
    target_compile_options(screamer_bindings PRIVATE /O2)
else()
    # NOTE: Do NOT add -ffast-math. The algorithms use NaN as a sentinel
    # for "warmup / not enough data", and -ffinite-math-only (implied by
    # -ffast-math) lets the optimizer dead-code those NaN return paths.
    # Tests fail silently because NaN gets folded to 0 or to leaked values.
    target_compile_options(screamer_bindings PRIVATE -O3 -flto)
    target_link_options(screamer_bindings PRIVATE -flto)
    # GCC fuses multiply-adds across statements by default (-ffp-contract=fast),
    # so the batch loop and the per-event step of one operator could round
    # differently after inlining. Pin it off for GCC (the Linux wheels). Clang
    # keeps its default (on): it fuses only within one expression, and the
    # batch and streaming paths of an operator share those expressions. Off
    # for clang too would cost up to 1.76x on recurrences such as EwMean (an
    # FMA has a shorter latency than a multiply and an add). Results can
    # therefore differ in the last bits between platforms.
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        target_compile_options(screamer_bindings PRIVATE -ffp-contract=off)
    endif()
    if(SCREAMER_NATIVE_ARCH)
        target_compile_options(screamer_bindings PRIVATE -march=native)
    endif()
endif()

# Install the module into the Python package directory
install(TARGETS screamer_bindings DESTINATION screamer)

# WASM-readiness gate: proves operator headers compile with no binding
# library present (see devtools/check_pure_headers.sh).
add_custom_target(check-pure-headers
    COMMAND bash ${CMAKE_SOURCE_DIR}/devtools/check_pure_headers.sh
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    COMMENT "Compile operator headers with no binding library (WASM-readiness gate)")
