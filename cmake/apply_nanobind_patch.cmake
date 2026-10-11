# Apply cmake/nanobind-surrogate-module.patch to a nanobind source tree, once.
#
# Usage: cmake -DSRC=<nanobind source dir> -DPATCH=<patch file> -P apply_nanobind_patch.cmake
#
# Idempotent: when the patch already applies in reverse, the tree is already
# patched and nothing happens. Fails with FATAL_ERROR when the patch neither
# applies nor is already applied, so a build never silently uses an unpatched
# nanobind.

find_package(Git REQUIRED)

get_filename_component(SRC "${SRC}" ABSOLUTE)
get_filename_component(PATCH "${PATCH}" ABSOLUTE)

# Keep git from walking up into an enclosing repository (for example screamer's
# own checkout around _local_deps/nanobind); without a .git in SRC, git apply
# then patches the files relative to SRC.
get_filename_component(_parent "${SRC}" DIRECTORY)
set(ENV{GIT_CEILING_DIRECTORIES} "${_parent}")

execute_process(
    COMMAND "${GIT_EXECUTABLE}" apply --reverse --check "${PATCH}"
    WORKING_DIRECTORY "${SRC}"
    RESULT_VARIABLE _already OUTPUT_QUIET ERROR_QUIET)
if(_already EQUAL 0)
    message(STATUS "nanobind patch already applied in ${SRC}")
    return()
endif()

execute_process(
    COMMAND "${GIT_EXECUTABLE}" apply "${PATCH}"
    WORKING_DIRECTORY "${SRC}"
    RESULT_VARIABLE _applied ERROR_VARIABLE _err)
if(NOT _applied EQUAL 0)
    message(FATAL_ERROR "Could not apply ${PATCH} to ${SRC}:\n${_err}")
endif()
message(STATUS "Applied ${PATCH} to ${SRC}")
