#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <tuple>
#include <iterator>
#include "screamer/common/eval_op.h"
#include "screamer/common/functor_iterator.h"
#include "screamer/my_functors.h"

namespace py = pybind11;

using namespace screamer;

// FunctorBase<Derived, N, M> is the prototype for a generalized streaming
// algorithm with N inputs and M outputs. ScreamerBase is essentially the
// 1-input/1-output specialization of this idea.
//
// All four quadrants of the dispatcher are now implemented:
//   1->1 (Plan B), N->1 (Plan D), 1->M (Plan C), N->M (Plan E).
// MyFunctor11 is bound as a minimal demo of the 1->1 case. MyFunctor22 is
// bound as a minimal stateful demo of the N->M case (state survives
// between calls, which is what makes 2->2 dispatch interesting beyond the
// stateless polar pair).

void init_bindings_myfunctors(py::module& m) {

    py::class_<MyFunctor11, screamer::EvalOp>(m, "MyFunctor11")
        .def(py::init<>())
        .def("__call__", &MyFunctor11::handle_input)
        .def("reset", &MyFunctor11::reset, "Reset to the initial state.");

    bind_functor_iterator<MyFunctor11>(m, "MyFunctorIterator11");

    py::class_<MyFunctor22, screamer::EvalOp>(m, "MyFunctor22")
        .def(py::init<>())
        .def("__call__", &MyFunctor22::handle_input)
        .def("reset", &MyFunctor22::reset, "Reset to the initial state.");

    // MyFunctor31 (N=3, M=1): the C++ dispatcher works, but the auto-test
    //   harness in tests/param_cases.py feeds 1 numpy array per class; for a
    //   3-input class it should feed 3 parallel arrays. Deferred along with
    //   the test-harness extension for N>2 inputs.
}
