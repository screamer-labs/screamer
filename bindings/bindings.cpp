// #define NB_DETAILED_ERROR_MESSAGES

#include <nanobind/nanobind.h>

namespace nb = nanobind;

// Function declarations of submodules.
//
// PHASE 1 (this checkpoint) builds a REDUCED module: only the two converted
// submodules below are declared and initialised. The other 10 submodules are
// converted in Phase 3; when they are, restore their declarations here and
// their init calls in NB_MODULE (see CMakeLists.txt for re-expanding the build).
void init_bindings_core(nb::module_& m);
void init_bindings_signal(nb::module_& m);
// void init_bindings_math(nb::module_& m);
// void init_bindings_rolling(nb::module_& m);
// void init_bindings_expanding(nb::module_& m);
// void init_bindings_ew(nb::module_& m);
// void init_bindings_preprocessing(nb::module_& m);
// void init_bindings_fin(nb::module_& m);
// void init_bindings_misc(nb::module_& m);
// void init_bindings_micro(nb::module_& m);
// void init_bindings_streams(nb::module_& m);
// void init_bindings_dag(nb::module_& m);

NB_MODULE(screamer_bindings, m) {
    init_bindings_core(m);
    init_bindings_signal(m);
    // init_bindings_math(m);
    // init_bindings_rolling(m);
    // init_bindings_expanding(m);
    // init_bindings_ew(m);
    // init_bindings_preprocessing(m);
    // init_bindings_fin(m);
    // init_bindings_misc(m);
    // init_bindings_micro(m);
    // init_bindings_streams(m);
    // init_bindings_dag(m);
}
