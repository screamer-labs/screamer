#include "screamer/common/base.h"
#include "screamer/common/async_generator.h"
#include "screamer/common/lazy_eval_iterator.h"

namespace screamer {

bool is_dag_node(const py::object& obj) {
    return py::hasattr(obj, "is_node") &&
           obj.attr("is_node").cast<bool>() == true;
}

py::object make_dag_functor_node(py::object self, py::object args_tuple) {
    py::object mod = py::module_::import("screamer.dag");
    return mod.attr("make_functor_node")(self, args_tuple);
}

py::object ScreamerBase::operator()(py::object obj) {
    if (can_cast_to_double(obj)) {
        double value = py::cast<double>(obj);
        return py::float_(process_scalar(value));
    }

    if (py::isinstance<py::list>(obj) || py::isinstance<py::tuple>(obj)) {
        py::sequence seq = py::reinterpret_borrow<py::sequence>(obj);
        reset();
        py::list out;
        for (auto item : seq) {
            out.append(py::float_(process_scalar(item.cast<double>())));
        }
        return out;
    }

    if (py::isinstance<py::array>(obj)) {
        // Container/rank preservation (Rule A): an ndarray input returns an
        // ndarray of the same shape, whatever its length. A length-1 array is a
        // time series of one, not a scalar; only an actual Python scalar (handled
        // above) returns a scalar. process_python_array handles size 0 (empty)
        // and size 1 correctly, so no length special-case is needed.
        py::array_t<double> double_array_t = py::cast<py::array_t<double>>(obj);
        return process_python_array(double_array_t);
    }

    if (py::isinstance<py::iterable>(obj)) {
        std::vector<py::object> sources{obj};       // a single iterable of scalars (n_in==1)
        return py::cast(LazyEvalIterator(py::cast(this), std::move(sources)));
    }

    if (is_async_generator(obj)) {
        return py::cast(LazyAsyncIterator(obj, py::cast(this)));
    }

    if (is_dag_node(obj)) {
        py::object self = py::cast(this);
        return make_dag_functor_node(self, py::make_tuple(obj));
    }

    auto type_str = std::string(py::str(py::type::of(obj)));
    std::ostringstream oss;
    oss << "Unsupported input type for call: [" << type_str << "]";
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

py::array_t<double> ScreamerBase::process_python_array(py::array_t<double> input_array) {
    py::buffer_info buf_info = input_array.request();

    if (buf_info.ndim < 1 || buf_info.itemsize != sizeof(double)) {
        throw std::runtime_error("Input array must have at least one dimension and contain doubles");
    }

    double* input_data = static_cast<double*>(buf_info.ptr);
    py::array_t<double> result(buf_info.shape);
    py::buffer_info result_buf = result.request();
    double* result_data = static_cast<double*>(result_buf.ptr);
    size_t size = buf_info.shape[0];

    if (size == 0) {
        return result;
    }

    if (buf_info.ndim == 1 && buf_info.strides[0] == sizeof(double)) {
        reset();
        process_array_no_stride(result_data, input_data, size);
        reset();
        return result;
    }

    size_t rest_size = 1;  
    for (int i = 1; i < buf_info.ndim; ++i) {
        rest_size *= buf_info.shape[i];
    }

    std::vector<size_t> input_strides(buf_info.ndim);
    std::vector<size_t> result_strides(buf_info.ndim);

    for (int i = 0; i < buf_info.ndim; ++i) {
        input_strides[i] = buf_info.strides[i] / sizeof(double);
        result_strides[i] = result_buf.strides[i] / sizeof(double);
    }

    std::vector<size_t> col_input_offsets(rest_size);
    std::vector<size_t> col_result_offsets(rest_size);

    for (size_t col = 0; col < rest_size; ++col) {
        size_t temp_col = col;
        size_t col_input_offset = 0;
        size_t col_result_offset = 0;

        for (int dim = buf_info.ndim - 1; dim > 0; --dim) {
            size_t index_in_dim = temp_col % buf_info.shape[dim];
            col_input_offset += index_in_dim * input_strides[dim];
            col_result_offset += index_in_dim * result_strides[dim];
            temp_col /= buf_info.shape[dim];
        }

        col_input_offsets[col] = col_input_offset;
        col_result_offsets[col] = col_result_offset;
    }

    for (size_t col = 0; col < rest_size; ++col) {
        size_t input_index = col_input_offsets[col];
        size_t result_index = col_result_offsets[col];

        reset();

        process_array_stride(
            &result_data[result_index], 
            result_strides[0],
            &input_data[input_index], 
            input_strides[0],
            size
        );
    }

    reset();
    return result;
}

} // namespace screamer
