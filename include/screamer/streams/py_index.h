#ifndef SCREAMER_STREAMS_PY_INDEX_H
#define SCREAMER_STREAMS_PY_INDEX_H

// The engine's index rule (screamer/dag/checked_index.h) for one Python index.
// Shared by the live push, advance and every Python iterator feed (PySource).

#include <cstdint>
#include <string>
#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include "screamer/dag/checked_index.h"

namespace screamer {
namespace nb = nanobind;

// NumPy scalar types the index and value rules need, looked up once.
struct NumpyTypes {
    PyObject* integer;
    PyObject* floating;
    PyObject* complex64;
    PyObject* clongdouble;
    PyObject* int64;        // the common NumPy integer, checked by type first
};
inline const NumpyTypes& numpy_types() {
    static const NumpyTypes* t = [] {
        nb::module_ np = nb::module_::import_("numpy");
        auto get = [&](const char* name) { return nb::object(np.attr(name)).release().ptr(); };
        return new NumpyTypes{get("integer"), get("floating"), get("complex64"),
                              get("clongdouble"), get("int64")};
    }();
    return *t;
}

// A Python int as an index; one past int64 is out of range, not an OverflowError.
inline std::int64_t long_index(PyObject* p) {
    int overflow = 0;
    long long v = PyLong_AsLongLongAndOverflow(p, &overflow);
    if (overflow) screamer::dag::throw_index_out_of_range();
    if (v == -1 && PyErr_Occurred()) throw nb::python_error();
    return v;
}

// One index given in Python: an int, a float (np.float64), a NumPy integer or
// floating scalar. A bool, str, bytes, None, Decimal, Fraction or any other
// object is a TypeError. The rule and the messages are the engine's
// (checked_index.h).
inline std::int64_t py_index(nb::handle index) {
    PyObject* p = index.ptr();
    if (PyLong_CheckExact(p)) return long_index(p);
    if (PyFloat_Check(p)) return screamer::dag::checked_index(PyFloat_AsDouble(p));
    if (PyBool_Check(p)) screamer::dag::throw_index_kind("bool");
    if (PyLong_Check(p)) return long_index(p);
    const NumpyTypes& np = numpy_types();
    const int integer = reinterpret_cast<PyObject*>(Py_TYPE(p)) == np.int64
        ? 1 : PyObject_IsInstance(p, np.integer);
    if (integer < 0) throw nb::python_error();
    if (integer) {
        PyObject* ix = PyNumber_Index(p);
        if (!ix) throw nb::python_error();
        nb::object hold = nb::steal(ix);
        return long_index(ix);
    }
    const int floating = PyObject_IsInstance(p, np.floating);
    if (floating < 0) throw nb::python_error();
    if (floating) {                                  // np.float32, np.float16
        double d = PyFloat_AsDouble(p);
        if (d == -1.0 && PyErr_Occurred()) throw nb::python_error();
        return screamer::dag::checked_index(d);
    }
    screamer::dag::throw_index_kind(nb::cast<std::string>(nb::type_name(index.type())));
}

}  // namespace screamer
#endif
