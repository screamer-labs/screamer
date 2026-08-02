# Standalone Emscripten build for screamer.wasm (Embind point-op runtime).
# This is NOT the Python extension build; it compiles the pure C++ op cores plus
# the generated Embind bindings into a single ES6 module (screamer.mjs + .wasm).
#
# Invoke via wasm/build-wasm.sh, which passes the repo root as SCREAMER_ROOT.
cmake_minimum_required(VERSION 3.16)
project(screamer_wasm CXX)

# Repo root: build-wasm.sh passes -DSCREAMER_ROOT=<repo>. Fall back to the
# parent of this directory so a bare `emcmake cmake -S wasm` still resolves.
if(NOT DEFINED SCREAMER_ROOT)
    get_filename_component(SCREAMER_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/.." ABSOLUTE)
endif()

# Pure C++ sources: everything the ops compile from, minus the two nanobind TUs.
# Glob common/ and detail/ (and dag/ if it ever grows .cpp files) then drop the
# nanobind translation units explicitly.
file(GLOB_RECURSE PURE_SOURCES
    "${SCREAMER_ROOT}/src/screamer/common/*.cpp"
    "${SCREAMER_ROOT}/src/screamer/detail/*.cpp"
    "${SCREAMER_ROOT}/src/screamer/dag/*.cpp")
list(REMOVE_ITEM PURE_SOURCES
    "${SCREAMER_ROOT}/src/screamer/common/dispatch.cpp"
    "${SCREAMER_ROOT}/src/screamer/common/async_generator.cpp")

message(STATUS "screamer.wasm PURE_SOURCES:")
foreach(s ${PURE_SOURCES})
    message(STATUS "  ${s}")
endforeach()

add_executable(screamer
    "${CMAKE_CURRENT_SOURCE_DIR}/generated/bindings_wasm.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/dag_bindings.cpp"
    ${PURE_SOURCES})

target_compile_features(screamer PRIVATE cxx_std_17)
target_include_directories(screamer PRIVATE
    "${SCREAMER_ROOT}/include"
    "${CMAKE_CURRENT_SOURCE_DIR}")  # so #include "embind_runtime.h" resolves

set(EMCC_LINK "-lembind -sMODULARIZE=1 -sEXPORT_ES6=1 -sENVIRONMENT=node \
-sFILESYSTEM=0 -sALLOW_MEMORY_GROWTH=1 -sMALLOC=emmalloc -sEXPORTED_RUNTIME_METHODS=['HEAPF64']")
set_target_properties(screamer PROPERTIES SUFFIX ".mjs" LINK_FLAGS "${EMCC_LINK} -Oz")
target_compile_options(screamer PRIVATE -Oz)
