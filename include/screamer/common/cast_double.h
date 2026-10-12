#ifndef SCREAMER_PYTHON_TOOLS_H
#define SCREAMER_PYTHON_TOOLS_H

#include <nanobind/nanobind.h>
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
    if (nb::try_cast<double>(x, d) || nb::hasattr(x, "ndim")) return {};
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
// must not outlive the container it views.
class EventItems {
public:
    EventItems(const std::vector<nb::object>& v) : vec_(&v), size_(v.size()) {}
    explicit EventItems(nb::list l) : list_(l.ptr()), size_(static_cast<std::size_t>(PyList_Size(l.ptr()))) {}
    std::size_t size() const { return size_; }
    nb::handle operator[](std::size_t i) const {
        return vec_ ? nb::handle(vec_->operator[](i).ptr()) : nb::handle(PyList_GetItem(list_, static_cast<Py_ssize_t>(i)));
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
        if (nb::try_cast<double>(cur, d)) break;
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
