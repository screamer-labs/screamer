#ifndef SCREAMER_PYTHON_TOOLS_H
#define SCREAMER_PYTHON_TOOLS_H

#include <nanobind/nanobind.h>
#include <string>
#include <cstddef>

namespace nb = nanobind;

namespace screamer {

// Type metadata from arbitrary user classes. A class may set __module__ to any
// object (None, an int, bytes), to a str that does not encode as UTF-8, or to a
// property that raises, so nb::type_name and an unchecked c_str() can crash on
// it. These helpers read type names safely and never leave an error set.

// The type's __name__ (e.g. "RollingMean", "str"), read from the type slot.
// Never formats __module__.
inline std::string type_short_name(nb::handle tp) {
    nb::object n = nb::steal(PyType_GetName(reinterpret_cast<PyTypeObject*>(tp.ptr())));
    const char* s = n.is_valid() ? PyUnicode_AsUTF8AndSize(n.ptr(), nullptr) : nullptr;
    if (!s) { PyErr_Clear(); return "object"; }
    return s;
}

// The type's __module__ as UTF-8, or nullptr when it is missing, raises, is
// not a str, or does not encode. `holder` keeps the returned string alive.
inline const char* type_module_utf8(nb::handle tp, nb::object& holder) {
    holder = nb::steal(PyObject_GetAttrString(tp.ptr(), "__module__"));
    if (!holder.is_valid()) { PyErr_Clear(); return nullptr; }
    if (!PyUnicode_Check(holder.ptr())) return nullptr;
    const char* s = PyUnicode_AsUTF8AndSize(holder.ptr(), nullptr);
    if (!s) PyErr_Clear();
    return s;
}

// Decide whether a Python object can be cast to double.
//
// IMPORTANT: do NOT introduce static-storage-duration variables here that
// call into Python (e.g. `static const auto x = nb::...`). On Windows those
// initialisers run during DLL load -- before the module's PyInit is called and
// before the GIL state for this module is set up. A Python C API call then
// trips PyEval_SaveThread's "GIL not held" assertion, which on Windows
// escalates to Py_FatalError -> __fastfail(0xC0000409) ->
// STATUS_STACK_BUFFER_OVERRUN at import. Linux/macOS dynamic-loader semantics
// happen to forgive this; Windows does not. String comparison on the type name
// is enough here and avoids the issue entirely.
//
// nanobind note: `nb::isinstance<nb::float_>` does NOT catch numpy scalar types
// (e.g. np.float64), exactly as pybind did not. Keep the type-name check so a
// numpy scalar dispatches as a scalar, not as an array / iterable.

inline bool is_numpy_numerical_scalar(const nb::object& obj) {
    static constexpr const char* kNumpyPrefix = "<class 'numpy.";
    static constexpr size_t kPrefixLen = 14;  // strlen of kNumpyPrefix

    // str(type(obj)) -> e.g. "<class 'numpy.float64'>".
    nb::str type_repr(nb::handle((PyObject*) Py_TYPE(obj.ptr())));
    const char* c = type_repr.c_str();  // nullptr if a hostile __module__ does not encode
    if (!c) { PyErr_Clear(); return false; }
    std::string type_str = c;

    if (type_str.compare(0, kPrefixLen, kNumpyPrefix) != 0) {
        return false;
    }

    const std::string suffix = type_str.substr(kPrefixLen);
    return suffix == "uint32'>"  || suffix == "uint64'>" ||
           suffix == "int32'>"   || suffix == "int64'>"  ||
           suffix == "float32'>" || suffix == "float64'>";
}

inline bool can_cast_to_double(const nb::object& obj) {
    if (nb::isinstance<nb::float_>(obj) ||
        nb::isinstance<nb::int_>(obj) ||
        nb::isinstance<nb::bool_>(obj)) {
        return true;
    }
    return is_numpy_numerical_scalar(obj);
}

}  // namespace screamer

#endif  // SCREAMER_PYTHON_TOOLS_H
