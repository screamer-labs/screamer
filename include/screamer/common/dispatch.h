#ifndef SCREAMER_DISPATCH_H
#define SCREAMER_DISPATCH_H

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>
#include <sstream>
#include "screamer/common/base.h"          // pure ScreamerBase
#include "screamer/common/cast_double.h"
#include "screamer/common/async_generator.h"

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
// which nanobind's generic `nb::ndarray<>` does NOT do for free.
inline double load_elem(const void* base, int64_t idx, const nb::dlpack::dtype& dt) {
    using C = nb::dlpack::dtype_code;
    if (dt.code == (uint8_t) C::Float) {
        if (dt.bits == 64) return ((const double*)  base)[idx];
        if (dt.bits == 32) return (double)((const float*) base)[idx];
        if (dt.bits == 16) return (double) half_bits_to_float(((const uint16_t*) base)[idx]);
        // A wider float (numpy longdouble / float128, 80- or 128-bit) never
        // reaches load_elem: nanobind's nb::ndarray<> cannot represent that
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
inline bool is_ndarray(nb::handle h) {
    nb::ndarray<> tmp;
    return nb::try_cast<nb::ndarray<>>(h, tmp, /*convert=*/false);
}

// True iff `h` is a rank>=1 numpy array whose dtype nanobind's `nb::ndarray<>`
// cannot represent (e.g. numpy longdouble / float128, an unsupported DLPack
// float width). Such an object is array-like (carries dtype/shape/ndim) yet
// is_ndarray() is false because try_cast<nb::ndarray<>> rejects the dtype, so
// it would otherwise be mis-routed to the iterable branch. The dispatch layer
// coerces these to float64 upstream (coerce_to_f64) so the normal ndarray path
// handles them. A 0-d array (ndim==0) is excluded: it behaves like a scalar.
inline bool is_unsupported_dtype_array(nb::handle h) {
    return !is_ndarray(h)
        && nb::hasattr(h, "dtype")
        && nb::hasattr(h, "shape")
        && nb::hasattr(h, "ndim")
        && nb::cast<int>(h.attr("ndim")) > 0;
}

// Replica of pybind11's `py::array_t<double, py::array::forcecast>`: a
// C-contiguous float64 numpy copy of `h`. numpy raises TypeError for a
// non-numeric dtype (e.g. object arrays), matching the old forcecast error
// surface. (Complex arrays never reach here: nanobind's nb::ndarray<> accepts
// them, so they stay on the normal path where load_elem raises.)
inline nb::object coerce_to_f64(nb::handle h) {
    nb::module_ np = nb::module_::import_("numpy");
    return np.attr("ascontiguousarray")(h, nb::arg("dtype") = np.attr("float64"));
}

// Coerce a single dispatch input: if it is an unsupported-dtype array, return
// its float64 coercion; otherwise return it unchanged. The fast path (float64
// and every other nb::ndarray<>-accepted dtype, scalars, iterables) is never
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
inline bool is_c_contiguous(const nb::ndarray<>& a) {
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

inline ContigDouble read_contig_double(const nb::ndarray<>& a) {
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

}  // namespace detail

// Relocated ScreamerBase dispatch (was ScreamerBase::operator() and
// ScreamerBase::process_python_array). Free functions taking the base by
// reference so one function pointer binds every ScreamerBase op as __call__.
nb::object screamer_call(ScreamerBase& self, nb::object obj);
nb::object process_python_array(ScreamerBase& self, nb::ndarray<> input);

}  // namespace screamer
#endif
