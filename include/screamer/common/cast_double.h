#ifndef SCREAMER_PYTHON_TOOLS_H
#define SCREAMER_PYTHON_TOOLS_H

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>

namespace py = pybind11;

namespace screamer {

// Decide whether a Python object can be cast to double.
//
// IMPORTANT: do NOT introduce static-storage-duration variables here that
// call into Python or pybind11 (e.g. `static const auto x = py::dtype::of<>()`).
// On Windows those initialisers run during DLL load -- before
// PyInit_screamer_bindings is called and before pybind11 has set up the
// GIL state for this module. The Python C API call then trips
// PyEval_SaveThread's "GIL not held" assertion, which on Windows escalates
// to Py_FatalError -> __fastfail(0xC0000409) -> STATUS_STACK_BUFFER_OVERRUN
// at import. Linux/macOS dynamic-loader semantics happen to forgive this;
// Windows does not. Three earlier dead-code variants of this function
// (`is_numpy_numerical_scalar` and v2, plus a static `numerical_dtypes`
// array initialised by `py::dtype::of<>()` calls) caused exactly that
// crash through v0.1.55. Lazy initialisation inside a function is the
// cure if such a table is ever needed -- but here string comparison on
// the type name is enough and avoids the issue entirely.

inline bool is_numpy_numerical_scalar(const py::object& obj) {
    static constexpr const char* kNumpyPrefix = "<class 'numpy.";
    static constexpr size_t kPrefixLen = 14;  // strlen of kNumpyPrefix

    auto type_str = std::string(py::str(py::type::of(obj)));

    if (type_str.compare(0, kPrefixLen, kNumpyPrefix) != 0) {
        return false;
    }

    const std::string suffix = type_str.substr(kPrefixLen);
    return suffix == "uint32'>"  || suffix == "uint64'>" ||
           suffix == "int32'>"   || suffix == "int64'>"  ||
           suffix == "float32'>" || suffix == "float64'>";
}

inline bool can_cast_to_double(const py::object& obj) {
    if (py::isinstance<py::float_>(obj) ||
        py::isinstance<py::int_>(obj) ||
        py::isinstance<py::bool_>(obj)) {
        return true;
    }
    return is_numpy_numerical_scalar(obj);
}

}  // namespace screamer

#endif  // SCREAMER_PYTHON_TOOLS_H
