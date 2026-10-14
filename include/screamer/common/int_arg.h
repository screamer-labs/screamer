#ifndef SCREAMER_COMMON_INT_ARG_H
#define SCREAMER_COMMON_INT_ARG_H

// Number operator settings in the Python binding.
//
// IntArg<T> is an integer setting (a window, a lag, a count). It converts like
// the built-in nanobind integer caster (an int, a NumPy integer, anything with
// __index__ or __int__ that is not a float), except that a str, bytes or
// bytearray, a bool, a NumPy bool and a 0-d NumPy bool array are a wrong kind:
// a string is never a number (R-a), so RollingMean("3") raises TypeError
// instead of parsing it, and a bool is not a number setting, so
// RollingMean(True) raises TypeError as the JavaScript factory does. FloatArg
// is a float setting with the same bool rule; a string already fails the float
// caster. Bindings use IntArg<T> in place of T and FloatArg in place of double
// in nb::init<...>; each converts implicitly. A caster does not know the
// operator or the setting, so its message is the tail ("must be an integer,
// got bool"); the Python class (screamer/_functor_params.py) names both.

#include <Python.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>
#include <cstring>
#include <string>
#include <vector>

namespace screamer {

// A NumPy type by name (bool_, ndarray), once NumPy is loaded (else null).
// Found once and kept: NumPy is never unloaded, and its types outlive every
// object of them.
inline PyObject* numpy_type(const char* attr, PyObject*& cache) {
    if (cache) return cache;
    PyObject* name = PyUnicode_FromString("numpy");
    PyObject* np = name ? PyImport_GetModule(name) : nullptr;
    Py_XDECREF(name);
    if (np) {
        PyObject* t = PyObject_GetAttrString(np, attr);
        Py_DECREF(np);
        if (t) cache = t;              // keeps the reference
    }
    PyErr_Clear();
    return cache;
}

// Whether `o` is a 0-d NumPy array of dtype bool, which converts to a number
// as a bool does.
inline bool is_bool_array_0d(PyObject* o, PyObject* ndarray) {
    if (!PyType_IsSubtype(Py_TYPE(o), reinterpret_cast<PyTypeObject*>(ndarray))) return false;
    bool yes = false;
    if (PyObject* nd = PyObject_GetAttrString(o, "ndim")) {
        if (PyLong_AsLong(nd) == 0) {
            if (PyObject* dt = PyObject_GetAttrString(o, "dtype")) {
                if (PyObject* kind = PyObject_GetAttrString(dt, "kind")) {
                    const char* k = PyUnicode_Check(kind) ? PyUnicode_AsUTF8AndSize(kind, nullptr) : nullptr;
                    yes = k && std::strcmp(k, "b") == 0;
                    Py_DECREF(kind);
                }
                Py_DECREF(dt);
            }
        }
        Py_DECREF(nd);
    }
    PyErr_Clear();
    return yes;
}

// "bool" for a Python bool, a NumPy bool or a 0-d NumPy bool array (the type
// name screamer/dag.py _setting uses), else null.
inline const char* bool_kind(PyObject* o) {
    if (PyBool_Check(o)) return "bool";
    if (PyLong_CheckExact(o) || PyFloat_Check(o)) return nullptr;
    static PyObject* np_bool = nullptr;
    static PyObject* np_ndarray = nullptr;
    PyObject* b = numpy_type("bool_", np_bool);
    if (!b) return nullptr;                       // NumPy is not loaded
    if (reinterpret_cast<PyObject*>(Py_TYPE(o)) == b) return "bool";
    PyObject* nd = numpy_type("ndarray", np_ndarray);
    return nd && is_bool_array_0d(o, nd) ? "bool" : nullptr;
}

template <class T>
struct IntArg {
    T value{};
    const char* text = nullptr;     // the type name of a refused string or bool, or null
    bool out_of_range = false;      // an integer that T cannot hold
    // The conversion runs inside nb::init's call, so this TypeError reaches
    // Python with its own message (a caster cannot raise: it is noexcept).
    // An integer outside T's range is a ValueError, which the Python class
    // turns into the range message of int_settings.h.
    operator T() const {
        if (text)
            throw nanobind::type_error(
                (std::string("must be an integer, got ") + text).c_str());
        if (out_of_range)
            throw nanobind::value_error("an integer setting is outside the range of int");
        return value;
    }
};

struct FloatArg {
    double value = 0.0;
    const char* text = nullptr;     // the type name of a refused bool, or null
    operator double() const {
        if (text)
            throw nanobind::type_error(
                (std::string("must be a number, got ") + text).c_str());
        return value;
    }
};

// A setting that is a sequence of numbers (MovingAverage's taps): a bool
// element is a wrong kind, as in a FloatArg.
struct FloatVecArg {
    std::vector<double> value;
    const char* text = nullptr;     // the type name of a refused element, or null
    operator std::vector<double>() const {
        if (text)
            throw nanobind::type_error(
                (std::string("must be a number or a sequence of numbers, got ") + text).c_str());
        return value;
    }
};

// The kind of the first bool element of a sequence, else null.
inline const char* bool_element_kind(PyObject* o) {
    if (!PySequence_Check(o) || PyUnicode_Check(o) || PyBytes_Check(o) || PyByteArray_Check(o))
        return nullptr;
    const Py_ssize_t n = PySequence_Size(o);
    if (n < 0) { PyErr_Clear(); return nullptr; }
    for (Py_ssize_t i = 0; i < n; ++i) {
        PyObject* item = PySequence_GetItem(o, i);
        if (!item) { PyErr_Clear(); return nullptr; }
        const char* b = bool_kind(item);
        Py_DECREF(item);
        if (b) return b;
    }
    return nullptr;
}

}  // namespace screamer

namespace nanobind { namespace detail {

template <class T>
struct type_caster<screamer::IntArg<T>> {
    NB_TYPE_CASTER(screamer::IntArg<T>, const_name("int"))

    bool from_python(handle src, uint32_t flags, cleanup_list* cleanup) noexcept {
        PyObject* o = src.ptr();
        if (PyUnicode_Check(o) || PyBytes_Check(o) || PyByteArray_Check(o)) {
            // Accepted here and refused by the conversion, with a clear message.
            value.text = PyUnicode_Check(o) ? (PyUnicode_CheckExact(o) ? "str" : "a str subclass")
                       : PyBytes_Check(o) ? "bytes" : "bytearray";
            return true;
        }
        if (const char* b = screamer::bool_kind(o)) {
            value.text = b;
            return true;
        }
        make_caster<T> inner;
        if (!inner.from_python(src, flags, cleanup)) {
            // An integer (a Python int, a NumPy integer) T cannot hold: taken
            // here and refused by the conversion, never wrapped.
            if (!PyFloat_Check(o) && PyIndex_Check(o)) {
                PyObject* ix = PyNumber_Index(o);
                if (ix) {
                    Py_DECREF(ix);
                    value.out_of_range = true;
                    return true;
                }
                PyErr_Clear();
            }
            return false;
        }
        value.value = inner.value;
        return true;
    }

    static handle from_cpp(const screamer::IntArg<T>& v, rv_policy policy,
                           cleanup_list* cleanup) noexcept {
        return make_caster<T>::from_cpp(v.value, policy, cleanup);
    }
};

template <>
struct type_caster<screamer::FloatArg> {
    NB_TYPE_CASTER(screamer::FloatArg, const_name("float"))

    bool from_python(handle src, uint32_t flags, cleanup_list* cleanup) noexcept {
        if (const char* b = screamer::bool_kind(src.ptr())) {
            value.text = b;          // refused by the conversion, with a clear message
            return true;
        }
        make_caster<double> inner;
        if (!inner.from_python(src, flags, cleanup)) return false;
        value.value = inner.value;
        return true;
    }

    static handle from_cpp(const screamer::FloatArg& v, rv_policy policy,
                           cleanup_list* cleanup) noexcept {
        return make_caster<double>::from_cpp(v.value, policy, cleanup);
    }
};

template <>
struct type_caster<screamer::FloatVecArg> {
    NB_TYPE_CASTER(screamer::FloatVecArg, make_caster<std::vector<double>>::Name)

    bool from_python(handle src, uint32_t flags, cleanup_list* cleanup) noexcept {
        if (const char* b = screamer::bool_element_kind(src.ptr())) {
            value.text = b;
            return true;
        }
        make_caster<std::vector<double>> inner;
        if (!inner.from_python(src, flags, cleanup)) return false;
        value.value = std::move(inner.value);
        return true;
    }

    static handle from_cpp(const screamer::FloatVecArg& v, rv_policy policy,
                           cleanup_list* cleanup) noexcept {
        return make_caster<std::vector<double>>::from_cpp(v.value, policy, cleanup);
    }
};

}}  // namespace nanobind::detail
#endif
