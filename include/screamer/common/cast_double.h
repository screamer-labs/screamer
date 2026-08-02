#ifndef SCREAMER_PYTHON_TOOLS_H
#define SCREAMER_PYTHON_TOOLS_H

#include <nanobind/nanobind.h>
#include <string>
#include <cstddef>

namespace nb = nanobind;

namespace screamer {

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
    std::string type_str = type_repr.c_str();

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
