cmake_minimum_required(VERSION 3.18)
project(screamer VERSION 0.1.54 LANGUAGES CXX)

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


# Find Python and Development libraries
find_package(Python3 COMPONENTS Interpreter Development.Module REQUIRED)

# Include FetchContent module
include(FetchContent)

# Define a directory for dependencies to avoid redundant downloads
set(FETCHCONTENT_BASE_DIR ${CMAKE_BINARY_DIR}/_deps)

# Path to locally cached dependencies (must be populated manually or during an online run)
set(LOCAL_CACHE_DIR ${CMAKE_SOURCE_DIR}/_local_deps)

# pybind11 v3.0.1 -- pinned because of pybind11 issue #5993:
# v3.0.2 regressed Windows + std::* container bindings (std::optional,
# std::vector, std::array) and crashes at import time with
# "PyEval_SaveThread: the GIL is released". We use std::optional
# extensively in the Ew* classes, std::vector and std::array in the
# functor templates -- so this hits us. v3.0.1 was confirmed working.
# v3.0.4 (latest at time of writing) still has the regression.
# Revisit when issue #5993 is fixed (PR #6006 was the proposed fix).
set(SCREAMER_PYBIND11_TAG "v3.0.1")

if(EXISTS "${LOCAL_CACHE_DIR}/pybind11")
    message(STATUS "Using locally cached pybind11 at ${LOCAL_CACHE_DIR}/pybind11")
    set(pybind11_SOURCE_DIR "${LOCAL_CACHE_DIR}/pybind11")
    add_subdirectory("${pybind11_SOURCE_DIR}" "${CMAKE_BINARY_DIR}/pybind11")
else()
    message(STATUS "Downloading pybind11 ${SCREAMER_PYBIND11_TAG} from GitHub")
    FetchContent_Declare(
        pybind11
        GIT_REPOSITORY https://github.com/pybind/pybind11.git
        GIT_TAG        ${SCREAMER_PYBIND11_TAG}
    )
    FetchContent_MakeAvailable(pybind11)
endif()


# ---------------------------------------------------------
# Create the Python module
# ---------------------------------------------------------

# Collect all .cpp files in the bindings directory
file(GLOB_RECURSE BINDING_SOURCES "bindings/*.cpp")

# Collect all source files in the src directory
file(GLOB_RECURSE SCREAMER_SOURCES "src/screamer/*.cpp")


pybind11_add_module(screamer_bindings MODULE ${BINDING_SOURCES} ${SCREAMER_SOURCES})
target_link_libraries(screamer_bindings PRIVATE pybind11::module)
target_include_directories(
    screamer_bindings PUBLIC
    "${CMAKE_SOURCE_DIR}/include"
    "${pybind11_INCLUDE_DIRS}"
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
