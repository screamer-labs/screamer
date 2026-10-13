#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>   // std::vector <-> list
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <vector>
#include <string>
#include "screamer/common/eval_op.h"
#include "screamer/common/int_settings.h"
#include "screamer/common/base.h"
#include "screamer/common/async_generator.h"
#include "screamer/common/lazy_eval_iterator.h"
#include "screamer/common/gc.h"
#include "screamer/common/shape.h"
#include "screamer/common/finite_settings.h"

namespace nb = nanobind;
using namespace nb::literals;

void init_bindings_core(nb::module_& m) {

    nb::class_<screamer::EvalOp>(m, "EvalOp")
        .def_prop_ro("num_inputs", &screamer::EvalOp::n_in)
        .def_prop_ro("num_outputs", &screamer::EvalOp::n_out);

    // The message for a float setting that is not finite (finite_settings.h),
    // asked by the Python class after its constructor raised ValueError.
    m.def("_finite_setting_error", [](const std::string& op, const std::vector<std::string>& names,
                                      const std::vector<std::vector<double>>& values) {
        return screamer::finite_setting_error(op, names, values, false);
    }, "op"_a, "names"_a, "values"_a);
    // The message for an integer setting below its smallest value
    // (int_settings.h), asked on the same error path.
    m.def("_int_setting_error", [](const std::string& op, const std::vector<std::string>& names,
                                   const std::vector<double>& values) {
        return screamer::int_setting_error(op, names, values);
    }, "op"_a, "names"_a, "values"_a);
    m.def("_int_setting_min", [](const std::string& op, const std::string& setting) -> nb::object {
        long long v = 0;
        if (!screamer::int_setting_min(op, setting, v)) return nb::none();
        return nb::int_(v);
    }, "op"_a, "setting"_a);

    // Test/engine helper: run one event through an op.
    m.def("_eval_op", [](screamer::EvalOp& op, const std::vector<double>& in) {
        if (in.size() != op.n_in()) {
            throw nb::value_error(("_eval_op: expected " + std::to_string(op.n_in()) + " inputs").c_str());
        }
        std::vector<double> out(op.n_out());
        op.eval(in.data(), out.data());
        return out;
    });

    nb::class_<screamer::ScreamerBase, screamer::EvalOp>(m, "ScreamerBase")
        .def("process_scalar", &screamer::ScreamerBase::process_scalar);

    // Both iterators take part in reference cycles (a source that stores its
    // stream), so they support the garbage collector (gc.h).
    nb::class_<screamer::LazyEvalIterator>(m, "LazyEvalIterator",
                                           nb::type_slots(screamer::gc_slots<screamer::LazyEvalIterator>))
        .def("__iter__", &screamer::LazyEvalIterator::__iter__,
             nb::rv_policy::reference_internal)
        .def("__next__", &screamer::LazyEvalIterator::__next__);

    nb::class_<screamer::LazyAsyncIterator>(m, "LazyAsyncIterator",
                                            nb::type_slots(screamer::gc_slots<screamer::LazyAsyncIterator>))
        .def("__aiter__", &screamer::LazyAsyncIterator::__aiter__,
             nb::rv_policy::reference_internal)
        .def("__anext__", &screamer::LazyAsyncIterator::__anext__)
        .def("_step", &screamer::LazyAsyncIterator::step)
        .def("_begin", &screamer::LazyAsyncIterator::begin)
        .def("_end", &screamer::LazyAsyncIterator::end);

    // Shape-contract helpers (include/screamer/common/shape.h), used by
    // screamer.shape.reshape, combine_latest and the tests.
    m.def("_layout_plan", [](const std::string& layout, std::vector<std::size_t> shape) {
        auto p = screamer::shape::plan_layout(layout, shape);
        return nb::make_tuple(p.out, p.source, p.dropped, p.shape);
    }, "layout"_a, "shape"_a);
    m.def("_broadcast_shape", [](const std::vector<std::vector<std::size_t>>& shapes) {
        std::vector<screamer::shape::View> views;
        for (const auto& s : shapes) views.push_back(screamer::shape::contiguous_view(s));
        return screamer::shape::broadcast(views).shape;
    }, "shapes"_a);
    m.def("_input_shape", [](const std::vector<std::vector<std::size_t>>& shapes, std::size_t n_in) {
        return screamer::shape::broadcast(screamer::shape::input_views(shapes, n_in)).shape;
    }, "shapes"_a, "n_in"_a);
}
