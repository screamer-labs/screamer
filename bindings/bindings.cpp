// #define NB_DETAILED_ERROR_MESSAGES

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <string>
#include <utility>
#include "screamer/common/base.h"
#include "screamer/common/cast_double.h"
#include "screamer/common/eval_op.h"

namespace nb = nanobind;

// The last __call__ overload of every operator: nanobind reaches it only when
// no other overload took the arguments, which is a wrong number of inputs (a
// ValueError naming the operator, rule A4) or a keyword the operator does not
// take (a TypeError, as Python names it). A call that works never reaches it.
//
// A one-input operator takes its input by position or by keyword ("value").
// Its __call__ keeps nanobind's "medium" dispatcher, the one its first
// overload has: a *args/**kwargs overload would switch every call to the
// slower "complex" dispatcher (about 9 ns per scalar call). So its count
// overload takes 0 to 7 positional arguments (nanobind's simple-call limit,
// with self), each defaulting to a sentinel, and a keyword nanobind cannot
// bind keeps nanobind's TypeError. Every other operator's __call__ takes
// *args already and gets one *args/**kwargs overload.
static std::string arg_name(nb::handle key) {
    nb::str k(key);
    const char* c = PyUnicode_AsUTF8AndSize(k.ptr(), nullptr);
    if (!c) { PyErr_Clear(); return "?"; }
    return c;
}

static nb::object wrong_call(screamer::EvalOp& self, const nb::args& args,
                             const nb::kwargs& kwargs) {
    const std::string who = screamer::op_display_name(nb::find(self));
    for (auto [key, unused] : kwargs)
        throw nb::type_error((who + "() got an unexpected keyword argument '" + arg_name(key) +
                              "'").c_str());
    screamer::throw_input_count(who, self.n_in(), args.size());
}

// The default of the count overload's parameters: a fresh object no caller
// passes. Kept for the life of the process.
static PyObject* count_sentinel() {
    static PyObject* s = PyObject_CallNoArgs(reinterpret_cast<PyObject*>(&PyBaseObject_Type));
    return s;
}

static nb::object wrong_count(screamer::EvalOp& self, nb::handle a0, nb::handle a1, nb::handle a2,
                              nb::handle a3, nb::handle a4, nb::handle a5, nb::handle a6) {
    std::size_t given = 0;
    for (nb::handle a : {a0, a1, a2, a3, a4, a5, a6}) given += a.ptr() != count_sentinel();
    const std::string who = screamer::op_display_name(nb::find(self));
    if (given == self.n_in())   // one input, by a keyword other than its own
        throw nb::type_error((who + "() takes its input by position or by its keyword").c_str());
    screamer::throw_input_count(who, self.n_in(), given);
}

// Adds the overloads above to every operator class of the module that
// defines __call__.
static void add_wrong_call_overloads(nb::module_& m) {
    nb::handle eval_op = m.attr("EvalOp");
    nb::handle base = m.attr("ScreamerBase");
    nb::handle none = nb::borrow(count_sentinel());
    nb::dict d = nb::borrow<nb::dict>(m.attr("__dict__"));
    for (auto [name, cls] : d) {
        if (!PyType_Check(cls.ptr()) || cls.is(eval_op) || cls.is(base)) continue;
        if (PyObject_IsSubclass(cls.ptr(), eval_op.ptr()) != 1) { PyErr_Clear(); continue; }
        PyObject* own = PyObject_GetAttrString(cls.ptr(), "__dict__");
        const bool has_call = own && PyMapping_HasKeyString(own, "__call__");
        Py_XDECREF(own);
        PyErr_Clear();
        if (!has_call) continue;
        if (PyObject_IsSubclass(cls.ptr(), base.ptr()) == 1) {
            nb::cpp_function_def(&wrong_count, nb::scope(cls), nb::name("__call__"),
                                 nb::is_method(), nb::arg("_0").none() = none,
                                 nb::arg("_1").none() = none, nb::arg("_2").none() = none,
                                 nb::arg("_3").none() = none, nb::arg("_4").none() = none,
                                 nb::arg("_5").none() = none, nb::arg("_6").none() = none,
                                 nb::sig("def __call__(self, *inputs) -> object"));
            continue;
        }
        PyErr_Clear();
        nb::cpp_function_def(&wrong_call, nb::scope(cls), nb::name("__call__"), nb::is_method(),
                             nb::sig("def __call__(self, *args, **kwargs) -> object"));
    }
}

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
    add_wrong_call_overloads(m);
}
