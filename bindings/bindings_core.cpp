#include <pybind11/pybind11.h>
#include <pybind11/stl.h> // Required for std::optional support
#include <vector>
#include <string>
#include "screamer/common/eval_op.h"
#include "screamer/common/base.h"
#include "screamer/common/iterator.h"
#include "screamer/common/async_generator.h"
#include "screamer/common/lazy_eval_iterator.h"

namespace py = pybind11;

void init_bindings_core(py::module& m) {

    py::class_<screamer::EvalOp>(m, "EvalOp")
        .def_property_readonly("num_inputs", &screamer::EvalOp::n_in)
        .def_property_readonly("num_outputs", &screamer::EvalOp::n_out);

    // Test/engine helper: run one event through an op.
    m.def("_eval_op", [](screamer::EvalOp& op, const std::vector<double>& in) {
        if (in.size() != op.n_in()) {
            throw py::value_error("_eval_op: expected " + std::to_string(op.n_in()) + " inputs");
        }
        std::vector<double> out(op.n_out());
        op.eval(in.data(), out.data());
        return out;
    });

    py::class_<screamer::ScreamerBase, screamer::EvalOp>(m, "ScreamerBase")
        .def("process_scalar", &screamer::ScreamerBase::process_scalar);

    py::class_<screamer::LazyEvalIterator>(m, "LazyEvalIterator")
        .def("__iter__", &screamer::LazyEvalIterator::__iter__,
             py::return_value_policy::reference_internal)
        .def("__next__", &screamer::LazyEvalIterator::__next__);

    py::class_<screamer::LazyIterator>(m, "LazyIterator")
        .def("__iter__", &screamer::LazyIterator::__iter__, py::return_value_policy::reference_internal)
        .def("__next__", &screamer::LazyIterator::__next__);

    py::class_<screamer::AnextAwaitable>(m, "AnextAwaitable")
        .def("__await__", &screamer::AnextAwaitable::__await__);

    py::class_<screamer::LazyAsyncIterator>(m, "LazyAsyncIterator")
        .def("__aiter__", &screamer::LazyAsyncIterator::__aiter__)
        .def("__anext__", &screamer::LazyAsyncIterator::__anext__);
}