// #define NB_DETAILED_ERROR_MESSAGES

#include <nanobind/nanobind.h>

namespace nb = nanobind;

// Function declarations of submodules.
void init_bindings_core(nb::module_& m);
void init_bindings_signal(nb::module_& m);
void init_bindings_math(nb::module_& m);
void init_bindings_rolling(nb::module_& m);
void init_bindings_expanding(nb::module_& m);
void init_bindings_ew(nb::module_& m);
void init_bindings_preprocessing(nb::module_& m);
void init_bindings_fin(nb::module_& m);
void init_bindings_misc(nb::module_& m);
void init_bindings_micro(nb::module_& m);
void init_bindings_streams(nb::module_& m);
void init_bindings_dag(nb::module_& m);

NB_MODULE(screamer_bindings, m) {
    init_bindings_core(m);
    init_bindings_signal(m);
    init_bindings_math(m);
    init_bindings_rolling(m);
    init_bindings_expanding(m);
    init_bindings_ew(m);
    init_bindings_preprocessing(m);
    init_bindings_fin(m);
    init_bindings_misc(m);
    init_bindings_micro(m);
    init_bindings_streams(m);
    init_bindings_dag(m);
}
