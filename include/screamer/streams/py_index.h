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

// The kind of index a Pipeline call, a live session or a stream merge has
// taken so far: plain numbers, or times of one kind and unit. A time index is
// read as int64 ticks of its own unit, as a batch array is, so numbers and
// times, or times of two units, would mix scales; the first index fixes the
// kind and a different one raises TypeError (rule 5). Shared by every input
// of the call, and by advance().
struct TimeUnit {
    static constexpr int kNone = -1, kDatetime = 0, kTimedelta = 1, kNumber = 2;
    int kind = kNone;
    int base = 0;           // NPY_DATETIMEUNIT, for a time kind
    int num = 0;
};

inline std::string index_kind_text(int kind, int base, int num) {
    if (kind == TimeUnit::kNumber) return "a number";
    static const char* const names[] = {"Y", "M", "W", "B", "D", "h", "m", "s", "ms",
                                        "us", "ns", "ps", "fs", "as", "generic"};
    std::string s = kind == TimeUnit::kDatetime ? "a datetime64[" : "a timedelta64[";
    if (num != 1) s += std::to_string(num);
    s += (base >= 0 && base < 15) ? names[base] : "?";
    return s + "]";
}

[[noreturn]] inline void throw_index_kind_change(const TimeUnit& had, int kind, int base, int num) {
    throw screamer::dag::IndexTypeError(
        "index is " + index_kind_text(kind, base, num) + ", but earlier indexes were " +
        index_kind_text(had.kind, had.base, had.num) + "; every index of one "
        "Pipeline call, live session or stream merge must have one kind and unit "
        "(convert them to one unit first)");
}

// Records a plain-number index in `unit`; one compare once the kind is set.
inline void note_number(TimeUnit* unit) {
    if (!unit || unit->kind == TimeUnit::kNumber) return;
    if (unit->kind != TimeUnit::kNone) throw_index_kind_change(*unit, TimeUnit::kNumber, 0, 0);
    unit->kind = TimeUnit::kNumber;
}

// Records a time index of `kind`, `base`, `num` in `unit`.
inline void note_time(TimeUnit* unit, int kind, int base, int num) {
    if (!unit) return;
    if (unit->kind == TimeUnit::kNone) { *unit = TimeUnit{kind, base, num}; return; }
    if (unit->kind != kind || unit->base != base || unit->num != num)
        throw_index_kind_change(*unit, kind, base, num);
}

// A np.datetime64 / np.timedelta64 index as its int64 ticks. NaT raises.
inline std::int64_t time_index(PyObject* p, int kind, TimeUnit* unit) {
    const NpyTimeScalar* s = reinterpret_cast<const NpyTimeScalar*>(p);
    if (s->ticks == INT64_MIN) screamer::dag::throw_index_nat();
    note_time(unit, kind, s->base, s->num);
    return s->ticks;
}

// The NPY_DATETIMEUNIT of a unit name ("ns", "s", ...), -1 if unknown.
inline int time_unit_base(const std::string& name) {
    static const char* const names[] = {"Y", "M", "W", "B", "D", "h", "m", "s", "ms",
                                        "us", "ns", "ps", "fs", "as", "generic"};
    for (int i = 0; i < 15; ++i) if (name == names[i]) return i;
    return -1;
}

// True for a time index: a np.datetime64 / np.timedelta64 scalar, or a pandas
// Timestamp / Timedelta / NaT (they carry asm8). Off the number paths only.
inline bool is_time_index(nb::handle index) {
    PyObject* p = index.ptr();
    const NumpyTypes& np = numpy_types();
    PyTypeObject* type = Py_TYPE(p);
    if (PyType_IsSubtype(type, reinterpret_cast<PyTypeObject*>(np.datetime64)) ||
        PyType_IsSubtype(type, reinterpret_cast<PyTypeObject*>(np.timedelta64)))
        return true;
    const int has = PyObject_HasAttrString(p, "asm8");
    return has == 1;
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
// given, holds the index kind of the call or session (TimeUnit); it is updated
// before the index is returned, so a caller that may still reject the event
// passes a copy and commits it afterwards.
inline std::int64_t py_index(nb::handle index, TimeUnit* unit = nullptr) {
    PyObject* p = index.ptr();
    if (PyLong_CheckExact(p)) { note_number(unit); return long_index(p); }
    if (PyFloat_Check(p)) { note_number(unit); return screamer::dag::checked_index(PyFloat_AsDouble(p)); }
    if (PyBool_Check(p)) screamer::dag::throw_index_kind("bool");
    if (PyLong_Check(p)) { note_number(unit); return long_index(p); }
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
        note_number(unit);
        return long_index(ix);
    }
    const int floating = PyObject_IsInstance(p, np.floating);
    if (floating < 0) throw nb::python_error();
    if (floating) {                                  // np.float32, np.float16
        double d = PyFloat_AsDouble(p);
        if (d == -1.0 && PyErr_Occurred()) throw nb::python_error();
        note_number(unit);
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

// One index of a float merge (a stream merge whose first index is a float):
// numbers of any value, read as double, under the same kinds as py_index. A
// time scalar is its int64 ticks (after a number it raises the kind error); a
// bool, str, bytes, None, Decimal, Fraction or any other object raises the
// index kind TypeError.
inline double py_float_index(nb::handle index, TimeUnit* unit = nullptr) {
    PyObject* p = index.ptr();
    if (PyFloat_Check(p)) { note_number(unit); return PyFloat_AsDouble(p); }
    if (PyBool_Check(p)) screamer::dag::throw_index_kind("bool");
    if (PyLong_Check(p)) {
        note_number(unit);
        const double d = PyLong_AsDouble(p);
        if (d == -1.0 && PyErr_Occurred()) {
            PyErr_Clear();
            screamer::dag::throw_index_out_of_range();
        }
        return d;
    }
    const NumpyTypes& np = numpy_types();
    PyTypeObject* type = Py_TYPE(p);
    // np.timedelta64 is a np.integer; py_index reads it as ticks.
    if (PyType_IsSubtype(type, reinterpret_cast<PyTypeObject*>(np.timedelta64)))
        return static_cast<double>(py_index(index, unit));
    const int integer = PyObject_IsInstance(p, np.integer);
    if (integer < 0) throw nb::python_error();
    const int floating = integer ? 0 : PyObject_IsInstance(p, np.floating);
    if (floating < 0) throw nb::python_error();
    if (integer || floating) {
        nb::object f = nb::steal(PyNumber_Float(p));
        if (!f.is_valid()) throw nb::python_error();
        note_number(unit);
        return PyFloat_AsDouble(f.ptr());
    }
    // A np.datetime64, or a pandas Timestamp or Timedelta: ticks, as py_index.
    if (is_time_index(index)) return static_cast<double>(py_index(index, unit));
    screamer::dag::throw_index_kind(nb::cast<std::string>(nb::type_name(index.type())));
}

}  // namespace screamer
#endif
