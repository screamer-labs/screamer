#ifndef SCREAMER_STREAMS_PY_INDEX_H
#define SCREAMER_STREAMS_PY_INDEX_H

// The engine's index rule (screamer/dag/checked_index.h) for one Python index.
// Shared by the live push, advance, the object-array batch index and every
// Python iterator feed (PySource).

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
    PyObject* datetime64;
    PyObject* timedelta64;  // a subclass of np.signedinteger
};
inline const NumpyTypes& numpy_types() {
    static const NumpyTypes* t = [] {
        nb::module_ np = nb::module_::import_("numpy");
        auto get = [&](const char* name) { return nb::object(np.attr(name)).release().ptr(); };
        return new NumpyTypes{get("integer"), get("floating"), get("complex64"),
                              get("clongdouble"), get("int64"), get("datetime64"),
                              get("timedelta64")};
    }();
    return *t;
}

// The layout of a np.datetime64 / np.timedelta64 scalar (NumPy's public
// PyDatetimeScalarObject / PyTimedeltaScalarObject, unchanged since NumPy 1.7):
// the int64 ticks, then the unit (an NPY_DATETIMEUNIT and a multiplier).
struct NpyTimeScalar {
    PyObject_HEAD
    std::int64_t ticks;     // NaT is INT64_MIN
    int base;               // NPY_DATETIMEUNIT
    int num;
};

// The unit of the datetime64 / timedelta64 index a feed has used so far. A
// feed reads its times as int64 ticks of their own unit, as a batch array
// does; a second unit within one feed would mix scales, so it raises.
struct TimeUnit {
    int kind = -1;          // -1 none yet, 0 datetime64, 1 timedelta64
    int base = 0;
    int num = 0;
};

inline std::string time_unit_text(int kind, int base, int num) {
    static const char* const names[] = {"Y", "M", "W", "B", "D", "h", "m", "s", "ms",
                                        "us", "ns", "ps", "fs", "as", "generic"};
    std::string s = kind == 0 ? "datetime64[" : "timedelta64[";
    if (num != 1) s += std::to_string(num);
    s += (base >= 0 && base < 15) ? names[base] : "?";
    return s + "]";
}

// A np.datetime64 / np.timedelta64 index as its int64 ticks. NaT raises.
inline std::int64_t time_index(PyObject* p, int kind, TimeUnit* unit) {
    const NpyTimeScalar* s = reinterpret_cast<const NpyTimeScalar*>(p);
    if (s->ticks == INT64_MIN) screamer::dag::throw_index_nat();
    if (unit) {
        if (unit->kind < 0) {
            *unit = TimeUnit{kind, s->base, s->num};
        } else if (unit->kind != kind || unit->base != s->base || unit->num != s->num) {
            throw screamer::dag::IndexTypeError(
                "index unit changed from " + time_unit_text(unit->kind, unit->base, unit->num) +
                " to " + time_unit_text(kind, s->base, s->num) +
                " within one feed; give every index of a feed the same unit");
        }
    }
    return s->ticks;
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
// floating scalar, or a time: np.datetime64 / np.timedelta64 (and pandas
// Timestamp / Timedelta, through their NumPy value) read as int64 ticks of
// their own unit, as a batch array of them is. A bool, str, bytes, None,
// Decimal, Fraction or any other object is a TypeError, and so is NaT. The
// rule and the messages are the engine's (checked_index.h). `unit`, when
// given, holds the feed's time unit (TimeUnit).
inline std::int64_t py_index(nb::handle index, TimeUnit* unit = nullptr) {
    PyObject* p = index.ptr();
    if (PyLong_CheckExact(p)) return long_index(p);
    if (PyFloat_Check(p)) return screamer::dag::checked_index(PyFloat_AsDouble(p));
    if (PyBool_Check(p)) screamer::dag::throw_index_kind("bool");
    if (PyLong_Check(p)) return long_index(p);
    const NumpyTypes& np = numpy_types();
    PyTypeObject* type = Py_TYPE(p);
    const bool int64 = reinterpret_cast<PyObject*>(type) == np.int64;
    const int integer = int64 ? 1 : PyObject_IsInstance(p, np.integer);
    if (integer < 0) throw nb::python_error();
    if (integer) {
        // np.timedelta64 is a np.integer too.
        if (!int64 && PyType_IsSubtype(type, reinterpret_cast<PyTypeObject*>(np.timedelta64)))
            return time_index(p, 1, unit);
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
    if (PyType_IsSubtype(type, reinterpret_cast<PyTypeObject*>(np.datetime64)))
        return time_index(p, 0, unit);
    // A pandas Timestamp, Timedelta or NaT: its NumPy value (asm8).
    PyObject* asm8 = PyObject_GetAttrString(p, "asm8");
    if (asm8) {
        nb::object hold = nb::steal(asm8);
        PyTypeObject* at = Py_TYPE(asm8);
        if (PyType_IsSubtype(at, reinterpret_cast<PyTypeObject*>(np.datetime64)))
            return time_index(asm8, 0, unit);
        if (PyType_IsSubtype(at, reinterpret_cast<PyTypeObject*>(np.timedelta64)))
            return time_index(asm8, 1, unit);
    } else {
        PyErr_Clear();
    }
    screamer::dag::throw_index_kind(nb::cast<std::string>(nb::type_name(index.type())));
}

}  // namespace screamer
#endif
