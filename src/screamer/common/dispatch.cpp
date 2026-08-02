#include "screamer/common/dispatch.h"
#include "screamer/common/lazy_eval_iterator.h"
#include <nanobind/stl/vector.h>
#include <stdexcept>
#include <sstream>
#include <vector>

namespace screamer {

bool is_dag_node(const nb::object& obj) {
    return nb::hasattr(obj, "is_node") &&
           nb::cast<bool>(obj.attr("is_node")) == true;
}

nb::object make_dag_functor_node(nb::object self, nb::object args_tuple) {
    nb::object mod = nb::module_::import_("screamer.dag");
    return mod.attr("make_functor_node")(self, args_tuple);
}

nb::object screamer_call(ScreamerBase& self, nb::object obj) {
    if (can_cast_to_double(obj)) {
        double value = nb::cast<double>(obj);
        return nb::float_(self.process_scalar(value));
    }

    if (nb::isinstance<nb::list>(obj) || nb::isinstance<nb::tuple>(obj)) {
        self.reset();
        nb::list out;
        for (nb::handle item : obj) {
            out.append(nb::float_(self.process_scalar(nb::cast<double>(item))));
        }
        return out;
    }

    if (detail::is_ndarray(obj)) {
        // Container/rank preservation (Rule A): an ndarray input returns an
        // ndarray of the same shape, whatever its length. A length-1 array is a
        // time series of one, not a scalar; only an actual scalar returns a
        // scalar.
        nb::ndarray<> arr = nb::cast<nb::ndarray<>>(obj);
        if (arr.ndim() == 0) {
            // Rank 0: a 0-d array carries a single sample with no time axis, so
            // it behaves like a scalar - one event in, one scalar out.
            nb::dlpack::dtype dt = arr.dtype();
            return nb::float_(self.process_scalar(detail::load_elem(arr.data(), 0, dt)));
        }
        return process_python_array(self, arr);
    }

    if (detail::is_unsupported_dtype_array(obj)) {
        // A numpy array whose dtype nanobind's nb::ndarray<> cannot represent
        // (e.g. longdouble/float128). Coerce to a contiguous float64 array so
        // the normal ndarray path handles it, instead of falling through to the
        // iterable branch below. Checked after is_ndarray so no supported dtype
        // pays for this.
        nb::ndarray<> arr = nb::cast<nb::ndarray<>>(detail::coerce_to_f64(obj));
        return process_python_array(self, arr);
    }

    if (nb::hasattr(obj, "__iter__")) {
        std::vector<nb::object> sources{ obj };      // a single iterable of scalars (n_in==1)
        return nb::cast(LazyEvalIterator(nb::find(self), std::move(sources)));
    }

    if (is_async_generator(obj)) {
        return nb::cast(LazyAsyncIterator(obj, nb::find(self)));
    }

    if (is_dag_node(obj)) {
        nb::object self_obj = nb::find(self);
        return make_dag_functor_node(self_obj, nb::make_tuple(obj));
    }

    nb::str type_repr(nb::handle((PyObject*) Py_TYPE(obj.ptr())));
    std::ostringstream oss;
    oss << "Unsupported input type for call: [" << type_repr.c_str() << "]";
    throw std::invalid_argument(oss.str());
}

nb::object process_python_array(ScreamerBase& self, nb::ndarray<> input) {
    size_t nd = input.ndim();

    std::vector<size_t> shape(nd);
    size_t total = 1;
    for (size_t i = 0; i < nd; ++i) { shape[i] = input.shape(i); total *= shape[i]; }

    // Materialise a C-contiguous double view of the input BEFORE allocating the
    // output buffer. read_contig_double / load_elem can throw on an
    // unsupported dtype; doing this first means no raw `new`-owned output
    // buffer is ever live across a throwing call, so nothing leaks.
    nb::dlpack::dtype dt = input.dtype();
    bool f64 = (dt.code == (uint8_t) nb::dlpack::dtype_code::Float && dt.bits == 64);
    detail::ContigDouble tmp;
    const double* in = nullptr;
    if (total > 0) {
        if (f64 && detail::is_c_contiguous(input)) {
            in = (const double*) input.data();
        } else {
            tmp = detail::read_contig_double(input);
            in = tmp.data.data();
        }
    }

    double* out = new double[total ? total : 1];
    if (total == 0) {
        return detail::make_owned_array(out, shape);
    }

    size_t size = shape[0];
    size_t rest = total / size;

    if (nd == 1) {
        self.reset();
        self.process_array_no_stride(out, in, size);
        self.reset();
    } else {
        for (size_t col = 0; col < rest; ++col) {
            self.reset();
            self.process_array_stride(out + col, rest, in + col, rest, size);
        }
        self.reset();
    }

    return detail::make_owned_array(out, shape);
}

}  // namespace screamer
