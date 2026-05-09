#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <tuple>
#include <iterator>
#include "screamer/common/functor_iterator.h"
#include "screamer/my_functors.h"

namespace py = pybind11;

using namespace screamer;

// FunctorBase<Derived, N, M> is the prototype for a generalized streaming
// algorithm with N inputs and M outputs. ScreamerBase is essentially the
// 1-input/1-output specialization of this idea.
//
// Plan B (this session): only MyFunctor11 (N=1,M=1) is bound. The dispatcher
// is fully implemented for that case.
//
// Plan D (next session): introduce a real N>1, M=1 indicator on top of this
// (e.g. RollingCorr) — the N-in/1-out dispatcher is already implemented, we
// just need a real algorithm and to teach the test harness to feed it N
// arrays. MyFunctor31 will be re-bound then.
//
// Plan C (after D): finish the M>1 dispatch handlers (handle_input_1i_Mo,
// handle_input_Ni_Mo). MyFunctor22 will be re-bound then.

void init_bindings_myfunctors(py::module& m) {

    py::class_<MyFunctor11>(m, "MyFunctor11")
        .def(py::init<>())
        .def("__call__", &MyFunctor11::handle_input)
        .def("reset", &MyFunctor11::reset, "Reset to the initial state.");

    bind_functor_iterator<MyFunctor11>(m, "MyFunctorIterator11");

    // MyFunctor22 (N=2, M=2): deferred to Plan C — handle_input throws
    //   "Unsupported functor type: N > 1, M > 1" at runtime today.
    //
    // MyFunctor31 (N=3, M=1): the C++ dispatcher works, but the auto-test
    //   harness in tests/param_cases.py feeds 1 numpy array per class; for a
    //   3-input class it should feed 3 parallel arrays. Deferred to Plan D
    //   along with the test-harness extension.
}
