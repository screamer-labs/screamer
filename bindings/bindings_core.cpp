#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>   // std::vector <-> list
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <cstdint>
#include <optional>
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
#include "screamer/common/json.h"
#include "screamer/serial/archive.h"
#include "screamer/serial/cbor.h"
#include "screamer/serial/envelope.h"
#include "screamer/serial/errors.h"

namespace nb = nanobind;
using namespace nb::literals;

namespace {

namespace json = screamer::json;
namespace serial = screamer::serial;

// A test object for the archives (tests/test_serial_core.py): every field call
// of serial/archive.h, a nested block with its own state() and a nested
// object saved through write_state/read_state as a nested operator is.
struct SerialDemoBlock {
    double a = -0.0;
    std::vector<double> buf{1.0, 2.0, 3.0};
    int count = 2;

    template <class Ar>
    void state(Ar& ar) {
        ar("a", a);
        ar("buf", buf);
        ar("count", count);
        ar.check(count >= 0 && static_cast<std::size_t>(count) <= buf.size(),
                 "count must be at most the buffer length");
    }
};

struct SerialDemoOp {
    static constexpr int kStateVersion = 1;
    double total = 10.0;
    long long seen = 4;

    template <class Ar>
    void state(Ar& ar) {
        ar("total", total);
        ar("seen", seen);
    }
    void write_state(json::Value& out) const {
        out = json::Value::object();
        out.set("version", json::Value::integer(kStateVersion));
        out.set("values", serial::save_state(*this));
    }
    void read_state(const json::Value& in, const std::string& path) {
        const json::Value* version = in.kind == json::Value::Kind::Object ? in.get("version") : nullptr;
        const json::Value* values = in.kind == json::Value::Kind::Object ? in.get("values") : nullptr;
        if (!version || version->kind != json::Value::Kind::Int || !values)
            throw serial::StateMismatch(path, "expected {\"version\": N, \"values\": {...}}");
        if (version->i != kStateVersion) throw serial::StateVersionError(path, version->i, kStateVersion);
        serial::load_state(*this, *values, path);
    }
};

struct SerialDemo {
    explicit SerialDemo(std::size_t k) : fixed(k, 0.0), k_(k) {}
    double x = 1.5;
    int n = 1;
    std::vector<double> v{1.0, 2.0};
    std::vector<double> fixed;
    SerialDemoBlock block;
    SerialDemoOp inner;
    bool flag = false;
    std::string label = "demo";
    std::optional<double> opt;
    std::size_t k_;

    template <class Ar>
    void state(Ar& ar) {
        ar("x", x);
        ar("n", n);
        ar("v", v);
        ar.fixed("fixed", fixed, k_);
        ar.object("block", block);
        ar.op("inner", inner);
        ar("flag", flag);
        ar("label", label);
        ar("opt", opt);
        ar.check(!(x < n), "x must not be below n");
    }
};

std::string utf8(nb::handle text) {
    Py_ssize_t n = 0;
    const char* u = PyUnicode_Check(text.ptr()) ? PyUnicode_AsUTF8AndSize(text.ptr(), &n) : nullptr;
    if (!u) {
        PyErr_Clear();
        throw nb::type_error("expected a str");
    }
    return std::string(u, static_cast<std::size_t>(n));
}

}  // namespace

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
                                   const std::vector<double>& values,
                                   const std::vector<std::string>& texts, bool alloc_failed) {
        return screamer::int_setting_error(op, names, values, texts, alloc_failed);
    }, "op"_a, "names"_a, "values"_a, "texts"_a, "alloc_failed"_a = false);
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

    // The errors of save and load (serial/errors.h), kept under private names
    // here (an upper-case public name of this module is taken for an operator
    // class); screamer exports them under their own names. A translator tries
    // the most recently registered type first, so the subclasses follow the
    // base.
    const auto hide = [&m](const char* name) {
        m.attr((std::string("_") + name).c_str()) = m.attr(name);
        nb::delattr(m, name);
    };
    auto save_error = nb::exception<serial::SaveError>(m, "SaveError", PyExc_ValueError);
    nb::exception<serial::StateVersionError>(m, "StateVersionError", save_error);
    nb::exception<serial::NotSaveable>(m, "NotSaveable", save_error);
    nb::exception<serial::StateMismatch>(m, "StateMismatch", save_error);
    for (const char* name : {"SaveError", "StateVersionError", "NotSaveable", "StateMismatch"}) hide(name);

    // Test hooks of the serial core (tests/test_serial_core.py): JSON text in,
    // the save layout (json::dump_save) out.
    m.def("_serial_dump_save", [](nb::handle text) {
        return json::dump_save(json::parse(utf8(text)));
    }, "text"_a);
    m.def("_serial_cbor_roundtrip", [](nb::handle text) {
        const std::vector<std::uint8_t> b = serial::cbor::encode(json::parse(utf8(text)));
        return json::dump_save(serial::cbor::decode(b.data(), b.size()));
    }, "text"_a);
    m.def("_serial_cbor_encode", [](nb::handle text) {
        const std::vector<std::uint8_t> b = serial::cbor::encode(json::parse(utf8(text)));
        return nb::bytes(reinterpret_cast<const char*>(b.data()), b.size());
    }, "text"_a);
    m.def("_serial_cbor_decode", [](nb::bytes data) {
        return json::dump_save(serial::cbor::decode(reinterpret_cast<const std::uint8_t*>(data.c_str()),
                                                    data.size()));
    }, "data"_a);
    m.def("_serial_wrap", [](nb::handle text) {
        return json::dump_save(serial::wrap(json::parse(utf8(text))));
    }, "text"_a);
    m.def("_serial_check_envelope", [](nb::handle text) {
        return json::dump_save(serial::unwrap(json::parse(utf8(text))));
    }, "text"_a);
    nb::class_<SerialDemo>(m, "_SerialDemo")
        .def(nb::init<std::size_t>(), "k"_a)
        .def("state", [](const SerialDemo& d) { return json::dump_save(serial::save_state(d)); })
        .def("set_state", [](SerialDemo& d, nb::handle text) {
            serial::load_state(d, json::parse(utf8(text)), "demo");
        }, "text"_a);
}
