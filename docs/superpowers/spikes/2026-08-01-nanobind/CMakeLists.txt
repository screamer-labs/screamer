cmake_minimum_required(VERSION 3.18)
project(nb_spike LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Find the interpreter first so nanobind builds against the venv we invoke it with.
find_package(Python 3.8 COMPONENTS Interpreter Development.Module Development.SABIModule REQUIRED)

include(FetchContent)
FetchContent_Declare(
  nanobind
  GIT_REPOSITORY https://github.com/wjakob/nanobind.git
  GIT_TAG        v2.9.2
)
FetchContent_MakeAvailable(nanobind)

# STABLE_ABI => abi3 wheel on CPython >= 3.12 (silently falls back on 3.11).
nanobind_add_module(nb_spike STABLE_ABI
  nb_spike.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/../src/screamer/detail/start_policy.cpp)

# Pull in the pybind-free screamer headers only.
target_include_directories(nb_spike PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../include)
