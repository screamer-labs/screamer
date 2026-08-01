#include "screamer/common/base.h"
#include "screamer/common/async_generator.h"
#include "screamer/common/lazy_eval_iterator.h"
#include <nanobind/stl/vector.h>
#include <stdexcept>

namespace screamer {

bool is_dag_node(const nb::object& obj) {
    return nb::hasattr(obj, "is_node") &&
           nb::cast<bool>(obj.attr("is_node")) == true;
}

nb::object make_dag_functor_node(nb::object self, nb::object args_tuple) {
    nb::object mod = nb::module_::import_("screamer.dag");
    return mod.attr("make_functor_node")(self, args_tuple);
}

nb::object ScreamerBase::operator()(nb::object obj) {
    if (can_cast_to_double(obj)) {
        double value = nb::cast<double>(obj);
        return nb::float_(process_scalar(value));
    }

    if (nb::isinstance<nb::list>(obj) || nb::isinstance<nb::tuple>(obj)) {
        reset();
        nb::list out;
        for (nb::handle item : obj) {
            out.append(nb::float_(process_scalar(nb::cast<double>(item))));
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
            return nb::float_(process_scalar(detail::load_elem(arr.data(), 0, dt)));
        }
        return process_python_array(arr);
    }

    if (nb::hasattr(obj, "__iter__")) {
        std::vector<nb::object> sources{ obj };      // a single iterable of scalars (n_in==1)
        return nb::cast(LazyEvalIterator(nb::find(*this), std::move(sources)));
    }

    if (is_async_generator(obj)) {
        return nb::cast(LazyAsyncIterator(obj, nb::find(*this)));
    }

    if (is_dag_node(obj)) {
        nb::object self = nb::find(*this);
        return make_dag_functor_node(self, nb::make_tuple(obj));
    }

    nb::str type_repr(nb::handle((PyObject*) Py_TYPE(obj.ptr())));
    std::ostringstream oss;
    oss << "Unsupported input type for call: [" << type_repr.c_str() << "]";
    throw std::invalid_argument(oss.str());
}

void ScreamerBase::process_array_no_stride(double* result_data, const double* input_data, size_t size) {
    for (size_t i = 0; i < size; i++) {
        result_data[i] = process_scalar(input_data[i]);
    }
}

void ScreamerBase::process_array_stride(
    double* result_data,
    size_t result_stride,
    const double* input_data,
    size_t input_stride,
    size_t size
) {
    size_t result_start = 0;
    size_t input_start = 0;

    for (size_t i = 0; i < size; i++) {
        result_data[result_start] = process_scalar(input_data[input_start]);
        result_start += result_stride;
        input_start += input_stride;
    }
}

nb::object ScreamerBase::process_python_array(nb::ndarray<> input) {
    size_t nd = input.ndim();

    std::vector<size_t> shape(nd);
    size_t total = 1;
    for (size_t i = 0; i < nd; ++i) { shape[i] = input.shape(i); total *= shape[i]; }

    double* out = new double[total ? total : 1];
    if (total == 0) {
        return detail::make_owned_array(out, shape);
    }

    size_t size = shape[0];
    size_t rest = total / size;

    // Materialise a C-contiguous double view of the input. The fast path avoids
    // the copy for an already-C-contiguous float64 array.
    nb::dlpack::dtype dt = input.dtype();
    bool f64 = (dt.code == (uint8_t) nb::dlpack::dtype_code::Float && dt.bits == 64);
    detail::ContigDouble tmp;
    const double* in;
    if (f64 && detail::is_c_contiguous(input)) {
        in = (const double*) input.data();
    } else {
        tmp = detail::read_contig_double(input);
        in = tmp.data.data();
    }

    if (nd == 1) {
        reset();
        process_array_no_stride(out, in, size);
        reset();
    } else {
        for (size_t col = 0; col < rest; ++col) {
            reset();
            process_array_stride(out + col, rest, in + col, rest, size);
        }
        reset();
    }

    return detail::make_owned_array(out, shape);
}

} // namespace screamer
