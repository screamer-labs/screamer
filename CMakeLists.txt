cmake_minimum_required(VERSION 3.18)
project(screamer VERSION 2.0.0 LANGUAGES CXX)

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
# platform. v2.9.2 is the tag validated by the migration spike; bump to a newer
# stable tag when re-validated. See docs/superpowers/plans/2026-08-01-nanobind-migration.md.
set(SCREAMER_NANOBIND_TAG "v2.9.2")

if(EXISTS "${LOCAL_CACHE_DIR}/nanobind")
    message(STATUS "Using locally cached nanobind at ${LOCAL_CACHE_DIR}/nanobind")
    add_subdirectory("${LOCAL_CACHE_DIR}/nanobind" "${CMAKE_BINARY_DIR}/nanobind")
else()
    message(STATUS "Downloading nanobind ${SCREAMER_NANOBIND_TAG} from GitHub")
    FetchContent_Declare(
        nanobind
        GIT_REPOSITORY https://github.com/wjakob/nanobind.git
        GIT_TAG        ${SCREAMER_NANOBIND_TAG}
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
    if(SCREAMER_NATIVE_ARCH)
        target_compile_options(screamer_bindings PRIVATE -march=native)
    endif()
endif()

# Install the module into the Python package directory
install(TARGETS screamer_bindings DESTINATION screamer)

# WASM-readiness gate: proves operator headers compile with no binding
# library present (see devtools/check_pure_headers.sh).
add_custom_target(check-pure-headers
    COMMAND ${CMAKE_COMMAND} -E env CXX=${CMAKE_CXX_COMPILER}
            bash ${CMAKE_SOURCE_DIR}/devtools/check_pure_headers.sh
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    COMMENT "Compile operator headers with no binding library (WASM-readiness gate)")
