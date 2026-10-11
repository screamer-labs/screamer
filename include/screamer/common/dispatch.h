#ifndef SCREAMER_DISPATCH_H
#define SCREAMER_DISPATCH_H

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/tuple.h>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>
#include <sstream>
#include "screamer/common/base.h"          // pure ScreamerBase
#include "screamer/common/functor_base.h"  // pure FunctorBase
#include "screamer/common/cast_double.h"
#include "screamer/common/shape.h"
#include "screamer/common/async_generator.h"
#include "screamer/common/lazy_eval_iterator.h"

namespace nb = nanobind;

namespace screamer {

// Returns true if obj is a screamer.dag.Node (duck-typed: has is_node True).
bool is_dag_node(const nb::object& obj);
// Build a graph node from a callable `self` and its argument objects.
nb::object make_dag_functor_node(nb::object self, nb::object args_tuple);

namespace detail {

// IEEE-754 binary16 (numpy float16 / "half") -> float, by bit manipulation.
// nanobind has no half-precision C++ type of its own (no `_Float16` /
// `nb::half`), so a half array's raw buffer is decoded directly from its
// 16-bit pattern rather than reinterpreted as some C++ scalar type. This is
// the standard sign/exponent/mantissa widening algorithm (subnormals included).
inline float half_bits_to_float(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t exp  = (h >> 10) & 0x1Fu;
    uint32_t mant = h & 0x3FFu;
    uint32_t bits;
    if (exp == 0) {
        if (mant == 0) {
            bits = sign;  // +/- 0
        } else {
            // Subnormal half -> normalize into a float exponent/mantissa.
            uint32_t e = 127 - 15 + 1;
            while ((mant & 0x400u) == 0) { mant <<= 1; --e; }
            mant &= 0x3FFu;
            bits = sign | (e << 23) | (mant << 13);
        }
    } else if (exp == 0x1Fu) {
        bits = sign | 0x7F800000u | (mant << 13);  // inf / nan
    } else {
        bits = sign | ((exp - 15u + 127u) << 23) | (mant << 13);
    }
    float f;
    std::memcpy(&f, &bits, sizeof(f));
    return f;
}

// Read one element of an arbitrary-dtype buffer at element index `idx`, coerced
// to double. Mirrors pybind's `py::array_t<double, forcecast>` dtype coercion,
// which nanobind's generic `nb::ndarray<nb::ro>` does NOT do for free.
inline double load_elem(const void* base, int64_t idx, const nb::dlpack::dtype& dt) {
    using C = nb::dlpack::dtype_code;
    if (dt.code == (uint8_t) C::Float) {
        if (dt.bits == 64) return ((const double*)  base)[idx];
        if (dt.bits == 32) return (double)((const float*) base)[idx];
        if (dt.bits == 16) return (double) half_bits_to_float(((const uint16_t*) base)[idx]);
        // A wider float (numpy longdouble / float128, 80- or 128-bit) never
        // reaches load_elem: nanobind's nb::ndarray<nb::ro> cannot represent that
        // dtype, so the dispatch layer coerces such an array to float64 via
        // numpy upstream (detail::coerce_if_unsupported_dtype_array) before any
        // element is read here. On a platform where long double == double (e.g.
        // Apple Silicon) it is already a 64-bit float, handled above.
    } else if (dt.code == (uint8_t) C::Int) {
        if (dt.bits == 64) return (double)((const int64_t*) base)[idx];
        if (dt.bits == 32) return (double)((const int32_t*) base)[idx];
        if (dt.bits == 16) return (double)((const int16_t*) base)[idx];
        if (dt.bits == 8)  return (double)((const int8_t*)  base)[idx];
    } else if (dt.code == (uint8_t) C::UInt) {
        if (dt.bits == 64) return (double)((const uint64_t*) base)[idx];
        if (dt.bits == 32) return (double)((const uint32_t*) base)[idx];
        if (dt.bits == 16) return (double)((const uint16_t*) base)[idx];
        if (dt.bits == 8)  return (double)((const uint8_t*)  base)[idx];
    } else if (dt.code == (uint8_t) C::Bool) {
        // numpy bool is a 1-byte array of 0/1. Old pybind11 forcecast coerced
        // it to double the same way; True/False -> 1.0/0.0.
        if (dt.bits == 8) return ((const uint8_t*) base)[idx] ? 1.0 : 0.0;
    }
    throw nb::type_error("Unsupported ndarray dtype; expected a numeric array.");
}

// Allocate a NEW numpy array that OWNS `data` via an owner capsule with a
// delete[] deleter, and return it as an nb::object. There is no owning
// ndarray(shape) constructor in nanobind; this is the required idiom. `data`
// must be a `new double[...]` C-contiguous buffer matching `shape`.
inline nb::object make_owned_array(double* data, const std::vector<size_t>& shape) {
    nb::capsule owner(data, [](void* p) noexcept { delete[] (double*) p; });
    return nb::cast(nb::ndarray<nb::numpy, double>(
        data, shape.size(), shape.data(), owner));
}

// True iff `h` is an ndarray-like object (numpy array), regardless of dtype /
// rank / strides. Replaces pybind's `py::isinstance<py::array>`. convert=false
// so it accepts existing arrays only (not sequences that could be converted).
//
// nanobind v2.9.2 to v2.12 read the type's __module__ inside the ndarray import
// with an unchecked c_str() and dereference NULL when it is not a str (fixed
// upstream in v2.13). A class with a non-str or non-encodable __module__ is
// never an array, so it returns false before reaching nanobind. Built-in
// (static) types, numpy.ndarray among them, always have a str __module__ and
// skip the lookup.
inline bool is_ndarray(nb::handle h) {
    PyTypeObject* tp = Py_TYPE(h.ptr());
    if (PyType_GetFlags(tp) & Py_TPFLAGS_HEAPTYPE) {
        nb::object holder;
        if (!type_module_utf8(nb::handle(reinterpret_cast<PyObject*>(tp)), holder)) return false;
    }
    nb::ndarray<nb::ro> tmp;
    return nb::try_cast<nb::ndarray<nb::ro>>(h, tmp, /*convert=*/false);
}

// True iff `h` is a rank>=1 numpy array whose dtype nanobind's `nb::ndarray<nb::ro>`
// cannot represent (e.g. numpy longdouble / float128, an unsupported DLPack
// float width). Such an object is array-like (carries dtype/shape/ndim) yet
// is_ndarray() is false because try_cast<nb::ndarray<nb::ro>> rejects the dtype, so
// it would otherwise be mis-routed to the iterable branch. The dispatch layer
// coerces these to float64 upstream (coerce_to_f64) so the normal ndarray path
// handles them. A 0-d array (ndim==0) is excluded: it behaves like a scalar.
inline bool is_unsupported_dtype_array(nb::handle h) {
    int nd = 0;
    return !is_ndarray(h)
        && nb::hasattr(h, "dtype")
        && nb::hasattr(h, "shape")
        && nb::hasattr(h, "ndim")
        && nb::try_cast<int>(h.attr("ndim"), nd) && nd > 0;
}

// Replica of pybind11's `py::array_t<double, py::array::forcecast>`: a
// C-contiguous float64 numpy copy of `h`. numpy raises TypeError for a
// non-numeric dtype (e.g. object arrays), matching the old forcecast error
// surface. (Complex arrays never reach here: nanobind's nb::ndarray<nb::ro> accepts
// them, so they stay on the normal path where load_elem raises.)
inline nb::object coerce_to_f64(nb::handle h) {
    nb::module_ np = nb::module_::import_("numpy");
    return np.attr("ascontiguousarray")(h, nb::arg("dtype") = np.attr("float64"));
}

// Coerce a single dispatch input: if it is an unsupported-dtype array, return
// its float64 coercion; otherwise return it unchanged. The fast path (float64
// and every other nb::ndarray<nb::ro>-accepted dtype, scalars, iterables) is never
// coerced.
inline nb::object coerce_if_unsupported_dtype_array(nb::object obj) {
    if (is_unsupported_dtype_array(obj)) return coerce_to_f64(obj);
    return obj;
}

// Coerce any unsupported-dtype array elements of an N-input argument pack.
// Preserves the fast path: returns `args` untouched (no rebuild) when nothing
// needs coercion.
inline nb::args coerce_args_if_unsupported(const nb::args& args) {
    bool any = false;
    for (nb::handle a : args) {
        if (is_unsupported_dtype_array(a)) { any = true; break; }
    }
    if (!any) return args;
    nb::list lst;
    for (nb::handle a : args) {
        lst.append(coerce_if_unsupported_dtype_array(nb::borrow<nb::object>(a)));
    }
    return nb::borrow<nb::args>(nb::steal(PyList_AsTuple(lst.ptr())));
}

// True iff `a` is C-contiguous (row-major, unit last stride).
inline bool is_c_contiguous(const nb::ndarray<nb::ro>& a) {
    size_t nd = a.ndim();
    if (nd == 0) return true;
    int64_t expected = 1;
    for (size_t d = nd; d-- > 0; ) {
        if (a.stride(d) != expected) return false;
        expected *= (int64_t) a.shape(d);
    }
    return true;
}

// A C-contiguous double copy of an ndarray of arbitrary dtype/stride, plus its
// shape. Mirrors pybind's forcecast to `py::array_t<double>`.
struct ContigDouble {
    std::vector<double> data;   // C-contiguous
    std::vector<size_t> shape;
    size_t ndim()  const { return shape.size(); }
    size_t total() const { size_t t = 1; for (size_t s : shape) t *= s; return t; }
    size_t time()  const { return shape.empty() ? 0 : shape[0]; }
    size_t cols()  const { size_t t = time(); return t ? total() / t : 0; }
};

inline ContigDouble read_contig_double(const nb::ndarray<nb::ro>& a) {
    ContigDouble r;
    size_t nd = a.ndim();
    r.shape.resize(nd);
    size_t total = 1;
    for (size_t i = 0; i < nd; ++i) { r.shape[i] = a.shape(i); total *= r.shape[i]; }
    r.data.resize(total);
    if (total == 0) return r;

    const void* base = a.data();
    nb::dlpack::dtype dt = a.dtype();

    // Fast path: already C-contiguous float64 -> straight copy.
    if (dt.code == (uint8_t) nb::dlpack::dtype_code::Float && dt.bits == 64
        && is_c_contiguous(a)) {
        std::memcpy(r.data.data(), base, total * sizeof(double));
        return r;
    }

    std::vector<int64_t> strides(nd);
    for (size_t i = 0; i < nd; ++i) strides[i] = a.stride(i);
    std::vector<size_t> idx(nd, 0);
    for (size_t lin = 0; lin < total; ++lin) {
        int64_t off = 0;
        for (size_t d = 0; d < nd; ++d) off += (int64_t) idx[d] * strides[d];
        r.data[lin] = load_elem(base, off, dt);
        // Odometer increment, C order (last axis fastest).
        for (size_t d = nd; d-- > 0; ) {
            if (++idx[d] < r.shape[d]) break;
            idx[d] = 0;
        }
    }
    return r;
}

// The N input fields of a call: one contiguous double buffer per argument (one
// buffer when a packed (T, *A, N) array holds all fields), each field's base
// pointer, and the broadcast plan that says where every column starts.
template <size_t N>
struct NInputs {
    std::vector<ContigDouble> buffers;
    std::array<double*, N> data{};
    shape::Plan plan;
};

// Read the argument arrays (any dtype and strides) and plan their broadcast
// (rules 1 and 4). Throws TypeError on a non-array argument and
// ValueError on shapes that do not broadcast.
template <size_t N>
inline NInputs<N> read_inputs(const std::vector<nb::handle>& arrays) {
    NInputs<N> r;
    r.buffers.reserve(arrays.size());
    std::vector<shape::Shape> shapes;
    for (nb::handle h : arrays) {
        if (!is_ndarray(h))
            throw nb::type_error("Incompatible input type, a mix of numpy arrays and other.");
        r.buffers.push_back(read_contig_double(nb::cast<nb::ndarray<nb::ro>>(h)));
        if (r.buffers.back().ndim() < 1)
            throw std::runtime_error("Input array must have at least one dimension");
        shapes.push_back(r.buffers.back().shape);
    }
    r.plan = shape::broadcast(shape::input_views(shapes, N));
    for (size_t i = 0; i < N; ++i)
        r.data[i] = r.buffers[arrays.size() == 1 ? 0 : i].data.data();
    return r;
}

// One packed array (rank 2 or more) as the single argument of an N-input call.
template <size_t N>
inline std::optional<NInputs<N>> maybe_read_packed(const nb::args& args) {
    if (args.size() != 1 || !is_ndarray(args[0])) return std::nullopt;
    if (nb::cast<nb::ndarray<nb::ro>>(args[0]).ndim() < 2) return std::nullopt;
    return read_inputs<N>({ args[0] });
}

template <size_t N>
inline NInputs<N> read_tuple_inputs(const nb::tuple& inputs) {
    std::vector<nb::handle> arrays;
    for (nb::handle h : inputs) arrays.push_back(h);
    return read_inputs<N>(arrays);
}

}  // namespace detail

// An argument that is not data: names the operator and the argument's type.
[[noreturn]] inline void throw_unsupported_input(nb::handle self, nb::handle x) {
    throw nb::type_error((op_display_name(self) + ": a " + type_short_name(x.type()) +
        " is not a valid input; pass numbers, arrays, lists or iterators").c_str());
}

// No routing matched the arguments of an N-input call. Names the first
// argument that is neither a number nor iterable; when every argument is one
// of those, the call mixed numbers with sequences.
[[noreturn]] inline void throw_unsupported_inputs(nb::handle self, nb::handle inputs) {
    std::string kinds;
    for (nb::handle x : inputs) {
        double d;
        if (!nb::try_cast<double>(x, d) && !nb::hasattr(x, "__iter__"))
            throw_unsupported_input(self, x);
        kinds += (kinds.empty() ? "" : ", ") + type_short_name(x.type());
    }
    throw nb::type_error((op_display_name(self) + ": cannot combine inputs of types (" + kinds +
        "); pass all numbers, all arrays, or all sequences").c_str());
}

// True when a pandas Series/DataFrame holds datetime64 or timedelta64 values.
inline bool is_time_dtype(nb::handle x) {
    auto kind_is_time = [](nb::handle dt) {
        nb::str k(dt.attr("kind"));
        const char* c = k.c_str();
        if (!c) { PyErr_Clear(); return false; }
        return c[0] == 'M' || c[0] == 'm';
    };
    if (nb::hasattr(x, "dtypes")) {
        nb::object d = x.attr("dtypes");
        if (nb::hasattr(d, "kind")) return kind_is_time(d);
        for (nb::handle e : d) if (nb::hasattr(e, "kind") && kind_is_time(e)) return true;
        return false;
    }
    return nb::hasattr(x, "dtype") && nb::hasattr(x.attr("dtype"), "kind") && kind_is_time(x.attr("dtype"));
}

// Normalizes one call argument before any routing. A pandas Series or
// DataFrame (detected by its type's module and a to_numpy method, without
// importing pandas) is read as its values and becomes a float64 NumPy array.
// A str, bytes, dict, set or frozenset is not data and raises TypeError at the
// call. Everything else is returned unchanged. The common inputs (float, int,
// list, tuple, array) leave through the first checks without a module lookup.
template <class S>
inline nb::object normalize_input(nb::handle x, const S& op) {
    PyObject* o = x.ptr();
    if (PyFloat_CheckExact(o) || PyLong_CheckExact(o) || PyList_CheckExact(o) ||
        PyTuple_CheckExact(o))
        return nb::borrow(x);
    const bool bad = PyUnicode_Check(o) || PyBytes_Check(o) || PyByteArray_Check(o) ||
                     PyMemoryView_Check(o) || PyDict_Check(o) || PyAnySet_Check(o);
    if (bad) throw_unsupported_input(nb::find(op), x);
    // pandas classes are heap types; a built-in (static) type, numpy.ndarray
    // among them, is never pandas and leaves without a module lookup. A heap
    // type's __module__ is read once, safely (it may be any object).
    if (!(PyType_GetFlags(Py_TYPE(o)) & Py_TPFLAGS_HEAPTYPE)) return nb::borrow(x);
    nb::object holder;
    const char* m = type_module_utf8(x.type(), holder);
    if (m && (std::strcmp(m, "pandas") == 0 || std::strncmp(m, "pandas.", 7) == 0) &&
        nb::hasattr(x, "to_numpy")) {
        if (is_time_dtype(x))
            throw nb::type_error((op_display_name(nb::find(op)) + ": a pandas object with a "
                "datetime or timedelta dtype is not a valid input; pass numeric values "
                "(convert timestamps yourself)").c_str());
        return x.attr("to_numpy")(nb::arg("dtype") = nb::borrow(reinterpret_cast<PyObject*>(&PyFloat_Type)));
    }
    return nb::borrow(x);
}

// Normalizes every argument of a call; returns args itself, without
// allocating, when none needed conversion.
template <class S>
inline nb::args normalize_args(const nb::args& args, const S& op) {
    const std::size_t n = args.size();
    nb::object out;  // created at the first changed argument
    bool changed = false;
    for (std::size_t i = 0; i < n; ++i) {
        nb::handle a = args[i];
        nb::object v = normalize_input(a, op);
        if (!changed && v.ptr() != a.ptr()) {
            changed = true;
            out = nb::list();
            for (std::size_t k = 0; k < i; ++k) nb::cast<nb::list>(out).append(args[k]);
        }
        if (changed) nb::cast<nb::list>(out).append(v);
    }
    if (!changed) return args;
    return nb::borrow<nb::args>(nb::tuple(nb::cast<nb::list>(out)));
}

// Relocated ScreamerBase dispatch (was ScreamerBase::operator() and
// ScreamerBase::process_python_array). Free functions taking the base by
// reference so one function pointer binds every ScreamerBase op as __call__.
nb::object screamer_call(ScreamerBase& self, nb::object obj);
nb::object process_python_array(ScreamerBase& self, nb::ndarray<nb::ro> input);

// ---------------------------------------------------------------------------
// Relocated FunctorBase dispatch. Was FunctorBase::handle_input* member
// templates; moved here as free templates parameterized on the operator type
// Op (reading Op::kN / Op::kM), so operator headers carry no nanobind. Behavior
// is byte-identical to the member versions; this is a pure relocation.
// ---------------------------------------------------------------------------

// Returns true iff h is a numpy array of rank >= 1 (a time series). A 0-d
// array is rank 0 (one sample, no time axis) and behaves like a scalar, so it
// is excluded here and falls through to the scalar cast.
inline bool is_series_array(nb::handle h) {
    return detail::is_ndarray(h) && nb::cast<int>(h.attr("ndim")) > 0;
}

// Returns true iff h is an iterable that is NOT a list, tuple, or array.
// Generators and iter(...) objects satisfy this; raw list/tuple do not.
inline bool is_lazy_iterable(nb::handle h) {
    return nb::hasattr(h, "__iter__")
        && !nb::isinstance<nb::list>(h)
        && !nb::isinstance<nb::tuple>(h)
        && !detail::is_ndarray(h);
}

namespace detail {
inline bool has_axis(nb::handle h) {
    int nd = 0;
    return nb::hasattr(h, "ndim") && nb::try_cast<int>(h.attr("ndim"), nd) && nd > 0;
}
}  // namespace detail

// True for a list of asset events: its first item is an array with an axis.
// Only a list is a sequence of events; a tuple never is. `fields` also accepts
// a first item that is a tuple or list of fields whose first field is such an
// array.
inline bool is_event_list(nb::handle h, bool fields) {
    if (!nb::isinstance<nb::list>(h)) return false;
    if (nb::len(h) == 0) return false;
    nb::object first = nb::steal(PySequence_GetItem(h.ptr(), 0));
    if (detail::has_axis(first)) return true;
    return fields && (nb::isinstance<nb::tuple>(first) || nb::isinstance<nb::list>(first)) &&
           nb::len(first) > 0 &&
           detail::has_axis(nb::steal(PySequence_GetItem(first.ptr(), 0)));
}

// A list given to a multi-input operator is a sequence of events, so each event
// must hold all n inputs. Names the reading and the fix.
[[noreturn]] inline void throw_event_arity(nb::handle self, std::size_t i, std::size_t m,
                                           std::size_t n) {
    const std::string name = op_display_name(self);
    const std::string ns = std::to_string(n);
    throw nb::value_error((name + " expects " + ns +
        " inputs; a list is read as a sequence of events, and event " + std::to_string(i) +
        " has " + std::to_string(m) + " value" + (m == 1 ? "" : "s") + " (expected " + ns +
        "). To pass " + ns + " separate inputs, call " + name + "()" +
        (n > 2 ? "(a, b, ...)" : "(a, b)") + ".").c_str());
}

// Checks event 0 of a list up front (an array event is read before the per-event
// loops run). Later events are checked by those loops as they convert.
inline void check_event_arity(nb::handle self, nb::handle list, std::size_t n) {
    if (!nb::isinstance<nb::list>(list) || nb::len(list) == 0) return;
    nb::handle ev = PyList_GetItem(list.ptr(), 0);
    std::size_t m;
    if (nb::isinstance<nb::tuple>(ev) || nb::isinstance<nb::list>(ev)) {
        m = nb::len(ev);
    } else if (detail::has_axis(ev)) {
        // An object with an axis but no integer shape or size is not an array;
        // the conversion later names its kind.
        nb::tuple shape;
        std::size_t last = 0;
        if (!nb::try_cast<nb::tuple>(ev.attr("shape"), shape) || shape.size() == 0 ||
            !nb::try_cast<std::size_t>(shape[shape.size() - 1], last))
            return;
        if (last == n) return;
        if (!nb::try_cast<std::size_t>(ev.attr("size"), m)) return;
    } else if (PyNumber_Check(ev.ptr())) {
        m = 1;
    } else {
        return;
    }
    if (m != n) throw_event_arity(self, 0, m, n);
}

// A one-input operator takes one array, and a list of arrays is a sequence of
// events. A tuple whose first item is an array is most likely a
// (values, index) stream, so it raises instead of guessing.
inline void reject_array_tuple(nb::handle h) {
    if (!nb::isinstance<nb::tuple>(h) || nb::len(h) == 0) return;
    if (!detail::has_axis(nb::steal(PySequence_GetItem(h.ptr(), 0)))) return;
    throw nb::type_error(
        "a tuple of arrays is not an input of a one-input operator. A (values, index) "
        "stream goes to a stream operator (Resample, CombineLatest) or a Pipeline; a "
        "compute operator takes the values, op(values), and the index stays as it was. "
        "Pass a list of arrays for a sequence of events.");
}

// Runs a list of events through LazyEvalIterator and returns the results as a
// list (an eager list in gives a list out), leaving the operator reset.
inline nb::object collect_events(nb::object owner, std::vector<nb::object> sources) {
    const std::size_t n = sources.size();
    nb::object it = nb::cast(LazyEvalIterator(
        std::make_unique<EventStepper>(owner, n, /*stream=*/false), std::move(sources)));
    PyObject* out = PySequence_List(it.ptr());
    nb::cast<EvalOp&>(owner).reset();
    if (!out) throw nb::python_error();
    return nb::steal(out);
}

// An eager list call is an independent run (decision D8): it starts from a
// reset operator and leaves it reset, like a batch call, also when it fails.
// reset() is not noexcept, and a destructor must not throw (it may run while
// an exception unwinds), so the closing reset swallows its own error.
template <class Op>
struct EagerRun {
    Op& op;
    explicit EagerRun(Op& o) : op(o) { op.reset(); }
    ~EagerRun() {
        try { op.reset(); } catch (...) {}
    }
    EagerRun(const EagerRun&) = delete;
    EagerRun& operator=(const EagerRun&) = delete;
};

// The N values of one event; `event` is its position, for the error message.
template <class Op>
typename Op::InputArray cast_to_array(Op& self, const nb::tuple& tuple, std::size_t event) {
    constexpr size_t N = Op::kN;
    if (tuple.size() != N) {
        throw nb::type_error("Tuple size does not match the number of expected inputs.");
    }
    typename Op::InputArray array;
    for (size_t i = 0; i < N; ++i) {
        if (!nb::try_cast<double>(tuple[i], array[i])) {
            const std::string who = op_display_name(nb::find(self));
            const std::string where = where_text("event", static_cast<std::ptrdiff_t>(event));
            throw_if_wrong_kind(who, tuple, where);
            throw_bad_shape(who, "item [" + std::to_string(i) + "] of " + where, tuple[i]);
        }
    }
    return array;
}

template <class Op>
nb::tuple args_to_tuple_n(const nb::args& args) {
    constexpr size_t N = Op::kN;

    if (args.size() == 1) { // a container of N

        nb::object arg = nb::borrow<nb::object>(args[0]);

        // we only support tuples and lists
        if (!(nb::isinstance<nb::list>(arg) || nb::isinstance<nb::tuple>(arg))) {
            throw nb::type_error("Unsupported single argument input type. Supported types are lists or tuples.");
        }

        // convert to tuple (a list is copied; nb::cast<nb::tuple> would reject it)
        nb::tuple inputs = nb::tuple(arg);

        // validate size
        if (inputs.size() != N) {
            throw nb::type_error("Wrong number of elements in the single argument input list / tuple.");
        }

        return inputs;
    }

    if (args.size() == N) {
        return nb::cast<nb::tuple>(args);
    }

    throw nb::type_error("Wrong number of arguments.");
}

// Eager parallel iteration over N iterables: pull one item from each per
// step, stop when any is exhausted. Returns a list of ResultTuple.
template <class Op>
nb::object eager_parallel(Op& self, const nb::tuple& inputs) {
    constexpr size_t N = Op::kN;
    for (nb::handle h : inputs) {
        if (!is_event_list(h, false)) continue;
        std::vector<nb::object> sources;
        for (nb::handle s : inputs) sources.push_back(nb::borrow<nb::object>(s));
        return collect_events(nb::find(self), std::move(sources));
    }
    std::array<nb::object, N> iters;
    for (size_t i = 0; i < N; ++i) iters[i] = nb::borrow<nb::object>(inputs[i]).attr("__iter__")();
    EagerRun<Op> run(self);
    std::vector<typename Op::ResultTuple> results;
    for (std::size_t event = 0;; ++event) {
        typename Op::InputArray array;
        bool stop = false;
        for (size_t i = 0; i < N; ++i) {
            nb::object item;
            try {
                item = iters[i].attr("__next__")();
            } catch (nb::python_error& e) {
                if (e.matches(PyExc_StopIteration)) { stop = true; break; }
                throw;
            }
            if (!nb::try_cast<double>(item, array[i]))
                throw_not_a_number(op_display_name(nb::find(self)), item,
                                   where_text("event", event, i));
        }
        if (stop) break;
        results.push_back(self.call(array));
    }
    return nb::cast(results);
}

// ---------------------------------------------------------
// ONE INPUT, ONE OUTPUT (numpy)
// ---------------------------------------------------------
template <class Op>
nb::object handle_input_1i_1o_numpy(Op& self, const nb::ndarray<nb::ro>& input) {
    detail::ContigDouble in = detail::read_contig_double(input);
    if (in.ndim() < 1) {
        throw std::runtime_error("Input array must have at least one dimension and contain doubles");
    }
    std::vector<size_t> shape = shape::consume_unit_field(in.shape);
    size_t total = in.total();
    double* out = new double[total ? total : 1];
    size_t size = in.time();
    size_t cols = in.cols();   // product of trailing dims = C-contiguous axis0 stride

    for (size_t col = 0; col < cols; ++col) {
        self.reset();
        size_t idx = col;
        for (size_t i = 0; i < size; ++i) {
            out[idx] = self.call({ in.data[idx] });
            idx += cols;
        }
    }
    self.reset();
    return detail::make_owned_array(out, shape);
}

// ---------------------------------------------------------
// MULTIPLE INPUTS, ONE OUTPUT (numpy)
// ---------------------------------------------------------
template <class Op>
nb::object handle_input_Ni_1o_numpy(Op& self, detail::NInputs<Op::kN>& in) {
    constexpr size_t N = Op::kN;
    const shape::Plan& p = in.plan;
    const size_t T = p.T, cols = p.columns;
    double* out = new double[T * cols ? T * cols : 1];
    std::array<int64_t, N> instride{};
    for (size_t i = 0; i < N; ++i) instride[i] = (int64_t) p.time_stride[i];
    for (size_t col = 0; col < cols; ++col) {
        std::array<size_t, N> inoff{};
        for (size_t i = 0; i < N; ++i) inoff[i] = p.offset[i * cols + col];
        self.reset();
        if (!self.process_columns(out, (std::ptrdiff_t) cols, in.data, instride, inoff, col, T)) {
            std::array<size_t, N> idx = inoff;
            size_t idx_out = col;
            typename Op::InputArray call_array;
            for (size_t t = 0; t < T; ++t) {
                for (size_t i = 0; i < N; ++i) { call_array[i] = in.data[i][idx[i]]; idx[i] += p.time_stride[i]; }
                out[idx_out] = self.call(call_array);
                idx_out += cols;
            }
        }
    }
    self.reset();
    return detail::make_owned_array(out, p.shape);
}

// ---------------------------------------------------------
// ONE INPUT, M>1 OUTPUTS (numpy)
// ---------------------------------------------------------
// For a 1-input array of shape (T, ...) the output has shape
// (T, ..., M): an extra trailing axis of size M is appended for the
// M outputs per time step. Memory is written contiguously per step.
template <class Op>
nb::object handle_input_1i_Mo_numpy(Op& self, const nb::ndarray<nb::ro>& input) {
    constexpr size_t M = Op::kM;
    detail::ContigDouble in = detail::read_contig_double(input);
    if (in.ndim() < 1) {
        throw std::runtime_error("Input array must have at least one dimension and contain doubles");
    }
    std::vector<size_t> shape = shape::consume_unit_field(in.shape);
    shape.push_back(M);
    size_t total = in.total();
    double* out = new double[total * M ? total * M : 1];
    size_t size = in.time();
    size_t cols = in.cols();

    for (size_t col = 0; col < cols; ++col) {
        self.reset();
        for (size_t i = 0; i < size; ++i) {
            size_t in_idx = i * cols + col;
            typename Op::ResultTuple results = self.call({ in.data[in_idx] });
            detail::write_tuple_to_memory(&out[in_idx * M], results);
        }
    }
    self.reset();
    return detail::make_owned_array(out, shape);
}

// ---------------------------------------------------------
// MULTIPLE INPUTS, MULTIPLE OUTPUTS (numpy)
// ---------------------------------------------------------
// The natural composition of the N->1 and 1->M rules:
//   - inputs broadcast over assets and pair column by column (from N->1)
//   - output shape = broadcast (T, *A) + (M,) (from 1->M)
template <class Op>
nb::object handle_input_Ni_Mo_numpy(Op& self, detail::NInputs<Op::kN>& in) {
    constexpr size_t N = Op::kN;
    constexpr size_t M = Op::kM;
    const shape::Plan& p = in.plan;
    const size_t T = p.T, cols = p.columns;
    std::vector<size_t> shape = p.shape;
    shape.push_back(M);
    double* out = new double[T * cols * M ? T * cols * M : 1];
    for (size_t col = 0; col < cols; ++col) {
        self.reset();
        std::array<size_t, N> idx{};
        for (size_t i = 0; i < N; ++i) idx[i] = p.offset[i * cols + col];
        typename Op::InputArray call_array;
        for (size_t t = 0; t < T; ++t) {
            for (size_t j = 0; j < N; ++j) { call_array[j] = in.data[j][idx[j]]; idx[j] += p.time_stride[j]; }
            detail::write_tuple_to_memory(&out[(t * cols + col) * M], self.call(call_array));
        }
    }
    self.reset();
    return detail::make_owned_array(out, shape);
}

// ---------------------------------------------------------
// ONE INPUT, M>1 OUTPUTS
// ---------------------------------------------------------
template <class Op>
nb::object functor_1i_Mo(Op& self, nb::object input) {
    // Case 1: Numpy array. Checked before the scalar cast so a length-1 array
    // is a time series of one (array in, array out - Rule A), not a scalar.
    if (is_series_array(input)) {
        return handle_input_1i_Mo_numpy(self, nb::cast<nb::ndarray<nb::ro>>(input));
    }

    // Case 1b: a numpy array whose dtype nanobind's nb::ndarray<nb::ro> cannot
    // represent (e.g. longdouble/float128). Coerce to float64 and process as
    // an array. Checked after is_series_array so no supported dtype pays.
    if (detail::is_unsupported_dtype_array(input)) {
        return handle_input_1i_Mo_numpy(self,
            nb::cast<nb::ndarray<nb::ro>>(detail::coerce_to_f64(input)));
    }

    // Case 2: Scalar input -> tuple of M floats (one streaming event)
    {
        double d;
        if (nb::try_cast<double>(input, d)) {
            typename Op::InputArray input_array = { d };
            return nb::cast(self.call(input_array));
        }
    }

    // Case 3: Iterable.
    // True lazy iterators (generators, iter(...)) stream via LazyEvalIterator.
    // Concrete list/tuple inputs are eager and return a list.
    if (nb::hasattr(input, "__iter__")) {
        if (is_lazy_iterable(input)) {
            std::vector<nb::object> sources{ nb::borrow<nb::object>(input) };
            return nb::cast(LazyEvalIterator(nb::find(self), std::move(sources)));
        }
        reject_array_tuple(input);
        if (is_event_list(input, false))
            return collect_events(nb::find(self), { nb::borrow<nb::object>(input) });
        // Eager path for list/tuple input.
        EagerRun<Op> run(self);
        std::vector<typename Op::ResultTuple> results;
        std::size_t event = 0;
        for (nb::handle item : input) {
            double d;
            if (!nb::try_cast<double>(item, d))
                throw_not_a_number(op_display_name(nb::find(self)), item, where_text("event", event));
            typename Op::InputArray input_array = { d };
            results.push_back(self.call(input_array));
            ++event;
        }
        return nb::cast(results);
    }

    throw_unsupported_input(nb::find(self), input);
}

// ---------------------------------------------------------
// ONE INPUT, ONE OUTPUT
// ---------------------------------------------------------
template <class Op>
nb::object functor_1i_1o(Op& self, nb::object input) {
    constexpr size_t N = Op::kN;

    // Case 1: Numpy array. Checked before the scalar cast so a length-1 array
    // is a time series of one (array in, array out - Rule A), not a scalar;
    // only an actual Python scalar returns a scalar.
    if (is_series_array(input)) {
        return handle_input_1i_1o_numpy(self, nb::cast<nb::ndarray<nb::ro>>(input));
    }

    // Case 1b: a numpy array whose dtype nanobind's nb::ndarray<nb::ro> cannot
    // represent (e.g. longdouble/float128). Coerce to float64 and process as
    // an array. Checked after is_series_array so no supported dtype pays.
    if (detail::is_unsupported_dtype_array(input)) {
        return handle_input_1i_1o_numpy(self,
            nb::cast<nb::ndarray<nb::ro>>(detail::coerce_to_f64(input)));
    }

    // Case 2: Scalar input (one streaming event)
    {
        double d;
        if (nb::try_cast<double>(input, d)) {
            typename Op::InputArray input_array = { d };
            return nb::cast(self.call(input_array));
        }
    }

    // Case 2b: a lazy iterator streams through LazyEvalIterator, as for every
    // other operator (lazy in, lazy out).
    if (is_lazy_iterable(input)) {
        std::vector<nb::object> sources{ input };
        return nb::cast(LazyEvalIterator(nb::find(self), std::move(sources)));
    }

    // Case 3: Iterable. A list of asset events runs like an iterator of them.
    reject_array_tuple(input);
    if (is_event_list(input, false))
        return collect_events(nb::find(self), { input });
    if (nb::hasattr(input, "__iter__")) {
        EagerRun<Op> run(self);
        std::vector<typename Op::ResultTuple> results;

        std::size_t event = 0;
        for (nb::handle item : input) {
            double d;
            if (nb::try_cast<double>(item, d)) {
                // Case 2.1: Scalar input item
                typename Op::InputArray input_array = { d };
                results.push_back(self.call(input_array));
            } else if (nb::isinstance<nb::tuple>(item)) {
                nb::tuple tuple = nb::borrow<nb::tuple>(item);
                if (tuple.size() == N) {
                    typename Op::InputArray input_array = cast_to_array<Op>(self, tuple, event);
                    results.push_back(self.call(input_array));
                } else {
                    throw nb::type_error("Invalid tuple size in iterable.");
                }
            } else {
                throw_not_a_number(op_display_name(nb::find(self)), item, where_text("event", event));
            }
            ++event;
        }

        return nb::cast(results);
    }

    // Case no match:
    throw_unsupported_input(nb::find(self), input);
}

// ---------------------------------------------------------
// MULTIPLE INPUTS, ONE OUTPUT
// ---------------------------------------------------------
template <class Op>
nb::object functor_Ni_1o(Op& self, const nb::args& raw_args) {
    constexpr size_t N = Op::kN;
    // Coerce any nanobind-unsupported-dtype array (e.g. longdouble/float128)
    // to float64 before the array-vs-iterable routing below, so it is
    // processed as an array. No-op (returns raw_args) for supported dtypes.
    nb::args args = detail::coerce_args_if_unsupported(raw_args);

    if (auto ins = detail::maybe_read_packed<N>(args)) {
        return handle_input_Ni_1o_numpy(self, *ins);
    }

    if (args.size() == 1) check_event_arity(nb::find(self), args[0], N);

    // A list of packed asset events, or of tuples of per-field events.
    if (args.size() == 1 && is_event_list(args[0], true))
        return collect_events(nb::find(self), { nb::borrow<nb::object>(args[0]) });

   // Case 1: we need to get his out of the way first.
   // A single argument, list/tuple with N-tuples inside: [ (1,2,3), (4,5,6), ...]
    if (args.size() == 1) {
        nb::object input = nb::borrow<nb::object>(args[0]);
        if (nb::isinstance<nb::list>(input) || nb::isinstance<nb::tuple>(input)) {
            EagerRun<Op> run(self);
            bool valid = true;
            std::vector<typename Op::ResultTuple> results;
            const bool is_list = nb::isinstance<nb::list>(input);
            std::size_t event_index = 0;
            for (nb::handle item : input) {
                // A list of events reads a list event like a tuple event.
                if (!(nb::isinstance<nb::tuple>(item) || (is_list && nb::isinstance<nb::list>(item)))) {
                    if (is_list && PyNumber_Check(item.ptr())) throw_event_arity(nb::find(self), event_index, 1, N);
                    valid = false;
                    break;
                }
                nb::tuple tuple = nb::tuple(nb::borrow(item));
                if (tuple.size() != N) {
                    if (is_list) throw_event_arity(nb::find(self), event_index, tuple.size(), N);
                    valid = false;
                    break;
                }
                typename Op::InputArray input_array =
                    cast_to_array<Op>(self, tuple, event_index);
                ++event_index;
                results.push_back(self.call(input_array));
            }
            if (valid) {
                return nb::cast(results);
            }
        }
    }

    // Case 1b: one lazy iterator whose events hold all N fields streams through
    // LazyEvalIterator.
    if (args.size() == 1 && is_lazy_iterable(args[0])) {
        std::vector<nb::object> sources{ nb::borrow<nb::object>(args[0]) };
        return nb::cast(LazyEvalIterator(nb::find(self), std::move(sources)));
    }

    // after this, now we handle cases where we have N arguments
    nb::tuple inputs = args_to_tuple_n<Op>(args);

    // Case 2: a tuple of N numpy arrays whose asset axes broadcast (nparray, ...).
    // Checked before the scalar cast so N length-1 arrays are a series of one
    // (array in, array out - Rule A), not N scalars collapsing to one scalar.
    if (is_series_array(inputs[0])) {
        auto ins = detail::read_tuple_inputs<N>(inputs);
        return handle_input_Ni_1o_numpy(self, ins);
    }

    // Case 3: a tuple of N scalar inputs: (0.3, 1.2, 4.0) -> one scalar event
    {
        typename Op::InputArray array;
        bool ok = true;
        for (size_t i = 0; i < N; ++i) {
            if (!nb::try_cast<double>(nb::borrow<nb::object>(inputs[i]), array[i])) { ok = false; break; }
        }
        if (ok) return nb::cast(self.call(array));
    }

    // Case 4: a tuple of N iterables: ( [...], [...], [...] )
    bool all_iterable = true;
    for (nb::handle input : inputs) {
        all_iterable = all_iterable && nb::hasattr(input, "__iter__");
        if (!all_iterable) break;
    }

    if (all_iterable) {
        // True lazy iterators (generators, iter(...)) stream via LazyEvalIterator.
        // Concrete list/tuple inputs are eager and return a list.
        bool all_lazy = true;
        for (nb::handle input : inputs) {
            all_lazy = all_lazy && is_lazy_iterable(input);
            if (!all_lazy) break;
        }
        if (all_lazy) {
            std::vector<nb::object> sources;
            for (nb::handle input : inputs) sources.push_back(nb::borrow<nb::object>(input));
            return nb::cast(LazyEvalIterator(nb::find(self), std::move(sources)));
        }
        // Eager path: iterate over each input in parallel.
        return eager_parallel(self, inputs);
    }

    // Case no match:
    throw_unsupported_inputs(nb::find(self), inputs);
}

// ---------------------------------------------------------
// MULTIPLE INPUTS, MULTIPLE OUTPUTS
// ---------------------------------------------------------
template <class Op>
nb::object functor_Ni_Mo(Op& self, const nb::args& raw_args) {
    constexpr size_t N = Op::kN;
    // Coerce any nanobind-unsupported-dtype array (e.g. longdouble/float128)
    // to float64 before the array-vs-iterable routing below, so it is
    // processed as an array. No-op (returns raw_args) for supported dtypes.
    nb::args args = detail::coerce_args_if_unsupported(raw_args);

    if (auto ins = detail::maybe_read_packed<N>(args)) {
        return handle_input_Ni_Mo_numpy(self, *ins);
    }

    if (args.size() == 1) check_event_arity(nb::find(self), args[0], N);

    // A list of packed asset events, or of tuples of per-field events.
    if (args.size() == 1 && is_event_list(args[0], true))
        return collect_events(nb::find(self), { nb::borrow<nb::object>(args[0]) });

    // Case 1: single argument, list/tuple of N-tuples
    // [(x0, y0), (x1, y1), ...] -> list of M-tuples
    if (args.size() == 1) {
        nb::object input = nb::borrow<nb::object>(args[0]);
        if (nb::isinstance<nb::list>(input) || nb::isinstance<nb::tuple>(input)) {
            EagerRun<Op> run(self);
            bool valid = true;
            std::vector<typename Op::ResultTuple> results;
            const bool is_list = nb::isinstance<nb::list>(input);
            std::size_t event_index = 0;
            for (nb::handle item : input) {
                // A list of events reads a list event like a tuple event.
                if (!(nb::isinstance<nb::tuple>(item) || (is_list && nb::isinstance<nb::list>(item)))) {
                    if (is_list && PyNumber_Check(item.ptr())) throw_event_arity(nb::find(self), event_index, 1, N);
                    valid = false;
                    break;
                }
                nb::tuple tuple = nb::tuple(nb::borrow(item));
                if (tuple.size() != N) {
                    if (is_list) throw_event_arity(nb::find(self), event_index, tuple.size(), N);
                    valid = false;
                    break;
                }
                typename Op::InputArray input_array =
                    cast_to_array<Op>(self, tuple, event_index);
                ++event_index;
                results.push_back(self.call(input_array));
            }
            if (valid) {
                return nb::cast(results);
            }
        }
    }

    // Case 1b: one lazy iterator whose events hold all N fields streams through
    // LazyEvalIterator.
    if (args.size() == 1 && is_lazy_iterable(args[0])) {
        std::vector<nb::object> sources{ nb::borrow<nb::object>(args[0]) };
        return nb::cast(LazyEvalIterator(nb::find(self), std::move(sources)));
    }

    nb::tuple inputs = args_to_tuple_n<Op>(args);

    // Case 2: tuple of N numpy arrays whose asset axes broadcast.
    if (is_series_array(inputs[0])) {
        auto ins = detail::read_tuple_inputs<N>(inputs);
        return handle_input_Ni_Mo_numpy(self, ins);
    }

    // Case 3: tuple of N scalars -> single M-tuple (one streaming event)
    {
        typename Op::InputArray array;
        bool ok = true;
        for (size_t i = 0; i < N; ++i) {
            if (!nb::try_cast<double>(nb::borrow<nb::object>(inputs[i]), array[i])) { ok = false; break; }
        }
        if (ok) return nb::cast(self.call(array));
    }

    // Case 4: tuple of N iterables.
    bool all_iterable = true;
    for (nb::handle input : inputs) {
        all_iterable = all_iterable && nb::hasattr(input, "__iter__");
        if (!all_iterable) break;
    }

    if (all_iterable) {
        bool all_lazy = true;
        for (nb::handle input : inputs) {
            all_lazy = all_lazy && is_lazy_iterable(input);
            if (!all_lazy) break;
        }
        if (all_lazy) {
            std::vector<nb::object> sources;
            for (nb::handle input : inputs) sources.push_back(nb::borrow<nb::object>(input));
            return nb::cast(LazyEvalIterator(nb::find(self), std::move(sources)));
        }
        // Eager path: iterate over each input in parallel.
        return eager_parallel(self, inputs);
    }

    throw_unsupported_inputs(nb::find(self), inputs);
}

// ---------------------------------------------------------
// Main dispatcher (entry point bound as __call__ for every FunctorBase op)
// ---------------------------------------------------------
template <class Op>
nb::object functor_call(Op& self, nb::args raw) {
    constexpr size_t N = Op::kN, M = Op::kM;
    nb::args args = normalize_args(raw, self);
    for (nb::handle a : args) {
        if (screamer::is_dag_node(nb::borrow<nb::object>(a)))
            return screamer::make_dag_functor_node(nb::find(self), nb::cast<nb::tuple>(args));
    }
    if constexpr (N == 1) { if (args.size() != 1) throw nb::type_error("Wrong number of in puts"); }
    // Async iterables, for every arity: N sources or one source of events.
    if (nb::object it = route_async(nb::find(self), args, N); it.is_valid()) return it;
    if constexpr (N == 1 && M == 1) return functor_1i_1o<Op>(self, nb::borrow<nb::object>(args[0]));
    else if constexpr (N > 1 && M == 1) return functor_Ni_1o<Op>(self, args);
    else if constexpr (N == 1 && M > 1) return functor_1i_Mo<Op>(self, nb::borrow<nb::object>(args[0]));
    else return functor_Ni_Mo<Op>(self, args);
}

}  // namespace screamer
#endif
