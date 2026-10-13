// #define NB_DETAILED_ERROR_MESSAGES

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <string>
#include "screamer/common/base.h"
#include "screamer/common/cast_double.h"
#include "screamer/common/eval_op.h"

namespace nb = nanobind;

// The last __call__ overload of every operator: nanobind reaches it only when
// no other overload took the arguments, which is a wrong number of inputs (a
// ValueError naming the operator, rule A4) or a keyword the operator does not
// take (a TypeError, as Python names it). Never on a call that works, so it
// costs those nothing. `input` is the keyword a one-input operator takes its
// input by ("value"), empty for an operator without one.
static nb::object wrong_call(screamer::EvalOp& self, const nb::args& args,
                             const nb::kwargs& kwargs, const std::string& input) {
    nb::object owner = nb::find(self);
    const std::string who = screamer::op_display_name(owner);
    for (auto [key, unused] : kwargs) {
        nb::str k(key);
        const char* c = PyUnicode_AsUTF8AndSize(k.ptr(), nullptr);
        if (!c) { PyErr_Clear(); c = "?"; }
        if (input != c)
            throw nb::type_error((who + "() got an unexpected keyword argument '" + c + "'").c_str());
        if (args.size() != 0)
            throw nb::type_error((who + "() got multiple values for argument '" + c + "'").c_str());
    }
    screamer::throw_input_count(who, self.n_in(), args.size() + kwargs.size());
}

// The keyword of a one-input operator's __call__, read from its nanobind
// signature "def __call__(self, value: object | None) -> object"; empty when
// the first overload takes *args.
static std::string call_keyword(nb::handle call) {
    nb::object sig = nb::getattr(call, "__nb_signature__", nb::none());
    if (sig.is_none() || nb::len(sig) == 0) return "";
    const std::string text = nb::cast<std::string>(nb::borrow<nb::tuple>(sig)[0][0]);
    const std::string head = "(self, ";
    const std::size_t at = text.find(head);
    if (at == std::string::npos) return "";
    const std::size_t begin = at + head.size();
    const std::size_t end = text.find_first_of(":,)", begin);
    const std::string name = text.substr(begin, end - begin);
    return name.empty() || name[0] == '*' ? "" : name;
}

// Adds wrong_call to every operator class of the module that defines __call__.
static void add_wrong_call_overloads(nb::module_& m) {
    nb::handle eval_op = m.attr("EvalOp");
    nb::handle base = m.attr("ScreamerBase");
    nb::dict d = nb::borrow<nb::dict>(m.attr("__dict__"));
    for (auto [name, cls] : d) {
        if (!PyType_Check(cls.ptr()) || cls.is(eval_op) || cls.is(base)) continue;
        if (PyObject_IsSubclass(cls.ptr(), eval_op.ptr()) != 1) { PyErr_Clear(); continue; }
        PyObject* own = PyObject_GetAttrString(cls.ptr(), "__dict__");
        const bool has_call = own && PyMapping_HasKeyString(own, "__call__");
        Py_XDECREF(own);
        PyErr_Clear();
        if (!has_call) continue;
        std::string input = call_keyword(cls.attr("__call__"));
        nb::cpp_function_def(
            [input](screamer::EvalOp& self, nb::args args, nb::kwargs kwargs) -> nb::object {
                return wrong_call(self, args, kwargs, input);
            },
            nb::scope(cls), nb::name("__call__"), nb::is_method(),
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
