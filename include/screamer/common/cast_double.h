#ifndef SCREAMER_PYTHON_TOOLS_H
#define SCREAMER_PYTHON_TOOLS_H

#include <nanobind/nanobind.h>
#include <limits>
#include <string>
#include <cstddef>
#include <cstring>
#include <unordered_set>
#include <vector>

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

// The operator's class name for error messages, e.g. "RollingMean".
inline std::string op_display_name(nb::handle self) {
    std::string name = type_short_name(self.type());
    if (auto dot = name.rfind('.'); dot != std::string::npos) name.erase(0, dot + 1);
    return name;
}

// ---- Values that are never numbers -----------------------------------------
//
// A time (a NumPy datetime64 or timedelta64 scalar, NaT among them; a pandas
// Timestamp, Timedelta or NaT; a Python datetime, date or timedelta) and a
// ctypes object are not data, although NumPy or the buffer protocol would read
// them as numbers: a datetime64 has __float__ (its tick count) and exports
// its 8 bytes as a buffer, and a ctypes array exports its items. Each is a
// wrong kind (TypeError) wherever a value is read. A ctypes buffer also has no
// strides, which nanobind 2.13's buffer import dereferences (see
// CMakeLists.txt), so no ctypes object may reach an ndarray cast.

// numpy.datetime64 and numpy.timedelta64, looked up once numpy is loaded
// (under the GIL; a missing numpy leaves them null and is retried).
struct TimeScalarTypes {
    PyTypeObject* datetime64 = nullptr;
    PyTypeObject* timedelta64 = nullptr;
    bool found = false;
};

inline const TimeScalarTypes& time_scalar_types() {
    static TimeScalarTypes t;
    if (!t.found) {
        PyObject* name = PyUnicode_FromString("numpy");
        PyObject* np = name ? PyImport_GetModule(name) : nullptr;   // only if loaded
        Py_XDECREF(name);
        if (np) {
            PyObject* dt = PyObject_GetAttrString(np, "datetime64");
            PyObject* td = PyObject_GetAttrString(np, "timedelta64");
            if (dt && td && PyType_Check(dt) && PyType_Check(td)) {
                // numpy keeps both types alive for the life of the process.
                t.datetime64 = reinterpret_cast<PyTypeObject*>(dt);
                t.timedelta64 = reinterpret_cast<PyTypeObject*>(td);
                t.found = true;
            }
            Py_XDECREF(dt);
            Py_XDECREF(td);
            Py_DECREF(np);
        }
        PyErr_Clear();
    }
    return t;
}

// A NumPy datetime64 or timedelta64 scalar (NaT included): two pointer
// compares, cheap enough for every value read.
inline bool is_numpy_time_scalar(PyObject* o) {
    const TimeScalarTypes& t = time_scalar_types();
    const PyTypeObject* tp = Py_TYPE(o);
    return t.found && (tp == t.datetime64 || tp == t.timedelta64);
}

// decimal.Decimal and fractions.Fraction: exact numbers that a float64 cast
// would round silently, so never read as values (TypeError; pass float(x)).
// Both modules are imported once, on the first value read (under the GIL).
struct ExactNumberTypes {
    PyTypeObject* decimal = nullptr;
    PyTypeObject* fraction = nullptr;
};

inline const ExactNumberTypes& exact_number_types() {
    static const ExactNumberTypes t = [] {
        ExactNumberTypes r;
        auto get = [](const char* mod, const char* name) -> PyTypeObject* {
            PyObject* m = PyImport_ImportModule(mod);
            PyObject* ty = m ? PyObject_GetAttrString(m, name) : nullptr;
            Py_XDECREF(m);
            if (!ty || !PyType_Check(ty)) { Py_XDECREF(ty); PyErr_Clear(); return nullptr; }
            return reinterpret_cast<PyTypeObject*>(ty);   // kept for the process
        };
        r.decimal = get("decimal", "Decimal");
        r.fraction = get("fractions", "Fraction");
        return r;
    }();
    return t;
}

// A Decimal or Fraction (or a subclass): two pointer compares for the usual
// value, a subtype check only for another class defined in Python.
inline bool is_exact_number(PyObject* o) {
    const ExactNumberTypes& t = exact_number_types();
    PyTypeObject* tp = Py_TYPE(o);
    if (tp == t.decimal || tp == t.fraction) return true;
    if (!(PyType_GetFlags(tp) & Py_TPFLAGS_HEAPTYPE)) return false;   // NumPy scalars, ...
    return (t.decimal && PyType_IsSubtype(tp, t.decimal)) ||
           (t.fraction && PyType_IsSubtype(tp, t.fraction));
}

// A ctypes object (an instance of _ctypes._CData: c_double, an array, a
// structure). Every such class is a heap type, so callers ask only for heap
// types. _ctypes not loaded means no ctypes object exists.
inline bool is_ctypes_object(PyObject* o) {
    static PyTypeObject* cdata = nullptr;   // kept for the life of the process
    if (!cdata) {
        PyObject* name = PyUnicode_FromString("_ctypes");
        PyObject* m = name ? PyImport_GetModule(name) : nullptr;
        Py_XDECREF(name);
        if (!m) { PyErr_Clear(); return false; }
        PyObject* arr = PyObject_GetAttrString(m, "Array");
        Py_DECREF(m);
        PyObject* base = arr ? PyObject_GetAttrString(arr, "__base__") : nullptr;
        Py_XDECREF(arr);
        if (!base || !PyType_Check(base)) { Py_XDECREF(base); PyErr_Clear(); return false; }
        cdata = reinterpret_cast<PyTypeObject*>(base);   // the reference is kept
    }
    return PyObject_TypeCheck(o, cdata);
}

// A pandas object that holds no array (a Timestamp, Timedelta, NaT, Period or
// Interval): its type's module is pandas and it has neither dtype nor dtypes.
inline bool is_pandas_module(const char* m) {
    return m && (std::strcmp(m, "pandas") == 0 || std::strncmp(m, "pandas.", 7) == 0);
}

// A Python datetime, date or timedelta (pandas Timestamp and Timedelta are
// subclasses), looked up once the datetime module is loaded.
inline bool is_python_time(PyObject* o) {
    static PyTypeObject* date = nullptr;
    static PyTypeObject* delta = nullptr;
    if (!date) {
        PyObject* name = PyUnicode_FromString("datetime");
        PyObject* m = name ? PyImport_GetModule(name) : nullptr;
        Py_XDECREF(name);
        if (!m) { PyErr_Clear(); return false; }
        PyObject* d = PyObject_GetAttrString(m, "date");
        PyObject* t = PyObject_GetAttrString(m, "timedelta");
        Py_DECREF(m);
        if (!d || !t || !PyType_Check(d) || !PyType_Check(t)) {
            Py_XDECREF(d); Py_XDECREF(t); PyErr_Clear(); return false;
        }
        date = reinterpret_cast<PyTypeObject*>(d);    // references kept
        delta = reinterpret_cast<PyTypeObject*>(t);
    }
    return PyObject_TypeCheck(o, date) || PyObject_TypeCheck(o, delta);
}

// True for a value of any kind above (a time or a ctypes object). Off the
// number fast paths only: it reads the type's module for a heap type.
inline bool is_foreign_value(PyObject* o) {
    if (is_numpy_time_scalar(o) || is_python_time(o) || is_exact_number(o)) return true;
    PyTypeObject* tp = Py_TYPE(o);
    if (!(PyType_GetFlags(tp) & Py_TPFLAGS_HEAPTYPE)) return false;
    if (is_ctypes_object(o)) return true;
    PyObject* mod = PyObject_GetAttrString(reinterpret_cast<PyObject*>(tp), "__module__");
    const char* m = (mod && PyUnicode_Check(mod)) ? PyUnicode_AsUTF8AndSize(mod, nullptr) : nullptr;
    const bool pandas_scalar = is_pandas_module(m) && !PyObject_HasAttrString(o, "dtype") &&
                               !PyObject_HasAttrString(o, "dtypes");
    Py_XDECREF(mod);
    PyErr_Clear();
    return pandas_scalar;
}

// A memoryview over a time or a ctypes object (its raw bytes).
inline bool is_foreign_view(PyObject* o) {
    if (!PyMemoryView_Check(o)) return false;
    PyObject* base = PyObject_GetAttrString(o, "obj");
    if (!base) { PyErr_Clear(); return false; }
    const bool r = is_foreign_value(base);
    Py_DECREF(base);
    return r;
}

// ---- Input count (rule A4: a wrong count is a ValueError) ------------------

// "RollingCorr expects 2 inputs, got 3", the text JavaScript's RangeError
// gives too. `alt` names another accepted form, e.g. ", or one packed (T, K,
// 5) array".
[[noreturn]] inline void throw_input_count(const std::string& who, std::size_t n, std::size_t got,
                                           const std::string& alt = "") {
    throw nb::value_error((who + " expects " + std::to_string(n) + " input" + (n == 1 ? "" : "s") +
                           alt + ", got " + std::to_string(got)).c_str());
}

// ---- Wrong-kind elements (rule A4: a wrong kind is a TypeError) ------------
//
// The conversions of user elements (a list item, an event, a field) are
// try_cast calls; these helpers run only after one failed, so the happy path
// does no extra work and every message is built on the error path.

// "event 3", "event 3 of input 1" or `prefix` alone when index < 0.
inline std::string where_text(const char* prefix, std::ptrdiff_t index = -1,
                              std::ptrdiff_t input = -1) {
    std::string s = prefix;
    if (index >= 0) s += " " + std::to_string(index);
    if (input >= 0) s += " of input " + std::to_string(input);
    return s;
}

// The kind of a value that is not a number, with its article: "an object",
// "a str". An exact int here failed only because it does not fit a double.
inline std::string kind_text(nb::handle x) {
    if (PyLong_CheckExact(x.ptr())) return "an int too large to convert to a float";
    std::string name = type_short_name(x.type());
    const bool vowel = !name.empty() && std::strchr("aeiouAEIOU", name[0]) != nullptr;
    return (vowel ? "an " : "a ") + name;
}

// "who: where is a T, expected <expected>".
[[noreturn]] inline void throw_wrong_kind(const std::string& who, const std::string& where,
                                          nb::handle item, const char* expected = "a number") {
    throw nb::type_error((who + ": " + where + " is " + kind_text(item) +
                          ", expected " + expected).c_str());
}

// The search below is bounded: each list or tuple is entered once (a list
// that holds itself, or one shared list repeated, is searched once), nesting
// stops at the depth of any array, and the whole search stops after
// kWrongKindBudget elements. Every input that reaches it was already walked
// once by the failed conversion, so this costs no more than that walk.
inline constexpr long kWrongKindBudget = 1000000;

struct WrongKindSearch {
    std::vector<std::size_t> path;      // list indices to the element found
    std::unordered_set<PyObject*> entered;   // containers already searched
    long budget = kWrongKindBudget;
    bool exhausted = false;
};

// The first element of `x`, depth first through lists and tuples, that is not
// a number, list, tuple or array-like (an object with ndim). Returns an
// invalid object when there is none (or the budget ran out, see `exhausted`).
inline nb::object find_wrong_kind(nb::handle x, WrongKindSearch& s, int depth = 0) {
    if (--s.budget < 0) { s.exhausted = true; return {}; }
    PyObject* o = x.ptr();
    if (PyList_Check(o) || PyTuple_Check(o)) {
        if (depth >= 64) return {};
        if (!s.entered.insert(o).second) return {};
        const Py_ssize_t n = PySequence_Size(o);
        for (Py_ssize_t i = 0; i < n; ++i) {
            nb::object item = nb::steal(PySequence_GetItem(o, i));
            if (!item.is_valid()) { PyErr_Clear(); return {}; }
            s.path.push_back(static_cast<std::size_t>(i));
            nb::object bad = find_wrong_kind(item, s, depth + 1);
            if (bad.is_valid() || s.exhausted) return bad;
            s.path.pop_back();
        }
        return {};
    }
    double d;
    if (x.is_none()) return {};                       // a missing value, NaN
    const bool text = PyUnicode_Check(o) || PyBytes_Check(o) || PyByteArray_Check(o);
    if (is_foreign_view(o)) return x.attr("obj");    // a time's or ctypes object's bytes
    if (!text && !is_foreign_value(o) &&
        (nb::try_cast<double>(x, d) || nb::hasattr(x, "ndim")))
        return {};
    return nb::borrow(x);
}

// Throws the wrong-kind TypeError for the first such element of `x` (see
// find_wrong_kind) and returns when there is none, so the caller raises its
// own error. `where` names x itself, e.g. "event 3"; a nested element reads
// "item [1][0] of event 3". When the search runs out of budget it returns
// too: nothing wrong was found, so the caller's shape or conversion error
// stands.
inline void throw_if_wrong_kind(const std::string& who, nb::handle x, const std::string& where) {
    WrongKindSearch s;
    nb::object bad = find_wrong_kind(x, s);
    const std::string self = where.empty() ? "the input" : where;
    if (s.exhausted || !bad.is_valid()) return;
    if (s.path.empty()) throw_wrong_kind(who, self, bad);
    std::string item = "item ";
    for (std::size_t i : s.path) item += "[" + std::to_string(i) + "]";
    throw_wrong_kind(who, where.empty() ? item : item + " of " + where, bad);
}

// The items of one event, one per source, read in place: either a Python list
// (the async path, which reads the awaited items without copying them) or a
// vector of objects (the sync iterator's buffer). It holds no references, so it
// must not outlive the container it views, and a list must not be mutated
// during the step (the async coroutine's local list stays untouched in normal
// use; a list that shrank anyway raises IndexError).
class EventItems {
public:
    EventItems(const std::vector<nb::object>& v) : vec_(&v), size_(v.size()) {}
    explicit EventItems(nb::list l) : list_(l.ptr()), size_(l.size()) {}
    std::size_t size() const { return size_; }
    nb::handle operator[](std::size_t i) const {
        if (vec_) return nb::handle((*vec_)[i].ptr());
        PyObject* item = PyList_GetItem(list_, static_cast<Py_ssize_t>(i));   // borrowed
        if (!item) nb::raise_python_error();
        return nb::handle(item);
    }
private:
    const std::vector<nb::object>* vec_ = nullptr;
    PyObject* list_ = nullptr;
    std::size_t size_;
};

// One failed event of a stream, `items` holding one item per source: names
// the first wrong-kind element, as "event 3" (one source) or "event 3 of
// input 1", and returns when there is none.
inline void throw_if_wrong_kind_event(const std::string& who, const EventItems& items,
                                      std::size_t event) {
    for (std::size_t i = 0; i < items.size(); ++i)
        throw_if_wrong_kind(who, items[i], where_text("event", static_cast<std::ptrdiff_t>(event),
            items.size() > 1 ? static_cast<std::ptrdiff_t>(i) : -1));
}

// The shape of a nested list/tuple or array-like as "(2, 3)", read along the
// first items, or "" when it is not regular enough to tell (an empty or
// self-containing list, an unreadable shape). Error paths only.
inline std::string shape_text(nb::handle x) {
    std::vector<std::size_t> dims;
    nb::object cur = nb::borrow(x);
    for (int depth = 0; depth < 32; ++depth) {
        PyObject* o = cur.ptr();
        if (PyList_Check(o) || PyTuple_Check(o)) {
            const Py_ssize_t n = PySequence_Size(o);
            dims.push_back(static_cast<std::size_t>(n));
            if (n <= 0) break;
            cur = nb::steal(PySequence_GetItem(o, 0));
            if (!cur.is_valid()) { PyErr_Clear(); return ""; }
            continue;
        }
        double d;
        if (nb::try_cast<double>(cur, d) || cur.is_none()) break;
        nb::tuple shp;
        if (!nb::hasattr(cur, "shape") || !nb::try_cast<nb::tuple>(cur.attr("shape"), shp)) return "";
        for (nb::handle h : shp) {
            std::size_t k;
            if (!nb::try_cast<std::size_t>(h, k)) return "";
            dims.push_back(k);
        }
        break;
    }
    if (dims.size() >= 32) return "";
    std::string s = "(";
    for (std::size_t i = 0; i < dims.size(); ++i) s += (i ? ", " : "") + std::to_string(dims[i]);
    return s + (dims.size() == 1 ? ",)" : ")");
}

// An element holds only numbers but has the wrong shape (rule A4: a shape
// mismatch is a ValueError): "who: where has shape (2,), expected a number".
[[noreturn]] inline void throw_bad_shape(const std::string& who, const std::string& where,
                                         nb::handle x, const char* expected = "a number") {
    const std::string shape = shape_text(x);
    throw nb::value_error((who + ": " + where +
        (shape.empty() ? std::string(" has an irregular shape") : " has shape " + shape) +
        ", expected " + expected).c_str());
}

// ---- Strings and missing values (R-a, R-b) ---------------------------------
//
// A string is never a number: str, bytes and bytearray (and NumPy's str_ and
// bytes_, their subclasses) are a wrong kind wherever a value is read, so a
// float64 conversion must never see one. None is a missing value, NaN. Both
// are decided by kind, never by an error message, and only off the number
// fast paths.

inline bool is_text(nb::handle x) {
    PyObject* o = x.ptr();
    return PyUnicode_Check(o) || PyBytes_Check(o) || PyByteArray_Check(o);
}

// A number, or None as NaN. The number cast comes first, so a number pays
// nothing extra; a string never converts (it has no __float__).
inline bool read_double(nb::handle h, double& d) {
    PyObject* o = h.ptr();
    if (PyFloat_CheckExact(o)) { d = PyFloat_AsDouble(o); return true; }
    if (is_numpy_time_scalar(o)) return false;      // a time is never a number
    if (!PyLong_CheckExact(o) && !PyFloat_Check(o) && is_exact_number(o))
        return false;                                // nor a Decimal or Fraction
    if (nb::try_cast<double>(h, d)) return true;
    if (h.is_none()) { d = std::numeric_limits<double>::quiet_NaN(); return true; }
    return false;
}

// The wrong-kind TypeError for a string value, "<who>: <where> is a str,
// expected a number"; `who` may be empty.
[[noreturn]] inline void throw_text_value(const std::string& who, nb::handle x,
                                          const std::string& where = "a value") {
    const std::string w = where + " is " + kind_text(x) + ", expected a number";
    throw nb::type_error((who.empty() ? w : who + ": " + w).c_str());
}

// Refuses a value that holds a string, before a float64 conversion: a string
// itself, an array (or pandas object) whose dtype kind is U or S, or an
// object array holding one (scanned element by element; this runs only for
// object dtypes, a slow path already). Any other value returns untouched.
// A list item's type known to hold no string: a NumPy scalar type other than
// str_ and bytes_ (is_text catches those, as subclasses of str and bytes) is
// one number of a fixed kind. The first item of such a type is checked in full
// and the type remembered for the rest of the call, so a list of NumPy scalars
// costs one type compare per item instead of two attribute lookups.
struct TextFreeTypes {
    PyTypeObject* generic = nullptr;     // numpy.generic, looked up once per call
    PyTypeObject* last = nullptr;        // the last NumPy scalar type seen
    bool looked_up = false;
};

inline void refuse_text(nb::handle x, const std::string& who, int depth, TextFreeTypes& seen) {
    if (is_text(x)) throw_text_value(who, x, depth ? "an element" : "a value");
    PyObject* o = x.ptr();
    // A time or a ctypes object is never a number either, nor a memoryview
    // of one (its raw bytes).
    if (!PyList_Check(o) && !PyTuple_Check(o) && !PyFloat_Check(o) && !PyLong_Check(o) &&
        o != Py_None && is_foreign_value(o))
        throw_text_value(who, x, depth ? "an element" : "a value");
    if (is_foreign_view(o)) throw_text_value(who, x.attr("obj"), depth ? "an element" : "a value");
    if (PyList_Check(o) || PyTuple_Check(o)) {          // nested lists, read in C
        if (depth >= 32) return;
        const bool list = PyList_Check(o);
        const Py_ssize_t n = list ? PyList_Size(o) : PyTuple_Size(o);
        for (Py_ssize_t i = 0; i < n; ++i) {
            PyObject* e = list ? PyList_GetItem(o, i) : PyTuple_GetItem(o, i);   // borrowed
            if (!e) throw nb::python_error();
            if (PyFloat_Check(e) || PyLong_Check(e) || e == Py_None || Py_TYPE(e) == seen.last) continue;
            refuse_text(nb::handle(e), who, depth + 1, seen);
            if (PyList_Check(e) || PyTuple_Check(e) || is_text(nb::handle(e))) continue;
            if (!seen.looked_up) {
                seen.looked_up = true;
                PyObject* np = PyImport_GetModule(nb::str("numpy").ptr());   // only if loaded
                if (np) {
                    PyObject* g = PyObject_GetAttrString(np, "generic");
                    Py_DECREF(np);
                    if (g && PyType_Check(g)) seen.generic = reinterpret_cast<PyTypeObject*>(g);
                    Py_XDECREF(g);   // numpy keeps the type alive
                }
                PyErr_Clear();
            }
            if (seen.generic && PyObject_TypeCheck(e, seen.generic)) seen.last = Py_TYPE(e);
        }
        return;
    }
    if (PyFloat_Check(o) || PyLong_Check(o) || o == Py_None) return;
    PyObject* dt = PyObject_GetAttrString(x.ptr(), "dtype");
    if (!dt) { PyErr_Clear(); return; }
    nb::object dtype = nb::steal(dt);
    PyObject* k = PyObject_GetAttrString(dt, "kind");
    if (!k) { PyErr_Clear(); return; }
    nb::object kind = nb::steal(k);
    const char* c = PyUnicode_Check(k) ? PyUnicode_AsUTF8AndSize(k, nullptr) : nullptr;
    if (!c) { PyErr_Clear(); return; }
    if (c[0] == 'c')
        throw nb::type_error(((who.empty() ? std::string() : who + ": ") +
            "a value must be a real number, got an array of complex values (dtype " +
            nb::cast<std::string>(nb::str(dtype)) + ")").c_str());
    if (c[0] == 'U' || c[0] == 'S')
        throw nb::type_error(((who.empty() ? std::string() : who + ": ") +
            "a value must be a number, got an array of strings (dtype " +
            nb::cast<std::string>(nb::str(dtype)) + ")").c_str());
    // A time is never a number (as a pandas object with a time dtype). The
    // ticks are one cast away: .astype("int64").
    if (c[0] == 'M' || c[0] == 'm')
        throw nb::type_error(((who.empty() ? std::string() : who + ": ") +
            "a value must be a number, got an array of times (dtype " +
            nb::cast<std::string>(nb::str(dtype)) + "); pass .astype(\"int64\") for "
            "its tick counts").c_str());
    if (c[0] != 'O') return;
    nb::module_ np = nb::module_::import_("numpy");
    nb::object flat = np.attr("asarray")(x, nb::arg("dtype") = nb::borrow(
        reinterpret_cast<PyObject*>(&PyBaseObject_Type))).attr("ravel")();
    for (nb::handle e : flat)
        if (is_text(e) || is_foreign_value(e.ptr())) throw_text_value(who, e, "an element");
}

// The float64 conversion of an object array (or pandas object) failed: names
// the first element that is not a number, "<who>: an element is a Timestamp,
// expected a number", so no NumPy or pandas text reaches the user. Returns
// when every element reads as a number, and the caller re-raises.
inline void throw_first_non_number(nb::handle x, const std::string& who) {
    nb::module_ np = nb::module_::import_("numpy");
    nb::object obj = nb::borrow(reinterpret_cast<PyObject*>(&PyBaseObject_Type));
    nb::object flat = nb::hasattr(x, "to_numpy")
        ? nb::object(x.attr("to_numpy")(nb::arg("dtype") = obj).attr("ravel")())
        : nb::object(np.attr("asarray")(x, nb::arg("dtype") = obj).attr("ravel")());
    for (nb::handle e : flat) {
        double d;
        if (read_double(e, d)) continue;
        const std::string w = "an element is " + kind_text(e) + ", expected a number";
        throw nb::type_error((who.empty() ? w : who + ": " + w).c_str());
    }
}

inline void refuse_text(nb::handle x, const std::string& who = "", int depth = 0) {
    TextFreeTypes seen;
    refuse_text(x, who, depth, seen);
}

// An element that should be a number is not one: a TypeError naming the first
// wrong-kind element inside it, or, when it holds only numbers, a ValueError
// giving its shape.
[[noreturn]] inline void throw_not_a_number(const std::string& who, nb::handle item,
                                            const std::string& where) {
    throw_if_wrong_kind(who, item, where);
    throw_bad_shape(who, where, item);
}

// True for the Python errors a failed conversion raises (TypeError,
// ValueError); others, such as MemoryError or KeyboardInterrupt, pass through
// unlabelled.
inline bool is_conversion_error(nb::python_error& e) {
    return e.matches(PyExc_TypeError) || e.matches(PyExc_ValueError);
}
inline bool is_conversion_error(const nb::builtin_exception& e) {
    return e.type() == nb::exception_type::type_error ||
           e.type() == nb::exception_type::value_error;
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
