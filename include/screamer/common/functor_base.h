#ifndef FUNCTOR_BASE_H
#define FUNCTOR_BASE_H

#include <cstddef>
#include <optional>
#include <tuple>
#include <type_traits>
#include <utility>
#include <pybind11/stl.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <iostream>
#include "screamer/common/base.h"

namespace screamer {

namespace detail {
    // Helper struct to create a tuple of M doubles
    template <size_t M, typename T = double, typename... Ts>
    struct TupleOfDoublesHelper {
        using type = typename TupleOfDoublesHelper<M - 1, T, T, Ts...>::type;
    };

    // Specialization for 0
    template <typename T, typename... Ts>
    struct TupleOfDoublesHelper<0, T, Ts...> {
        using type = std::tuple<Ts...>;
    };

    // Alias for easier use
    template <size_t M>
    using TupleOfDoubles = typename TupleOfDoublesHelper<M>::type;


    inline size_t numpy_num_cols(const py::buffer_info& buf_info) {
        size_t num_cols = 1;
        for (int i = 1; i < buf_info.ndim; ++i) {
            num_cols *= buf_info.shape[i];
        }
        return num_cols;
    }


    inline size_t numpy_col_start_pos(const size_t column, const py::buffer_info& buf_info) {
        size_t start_pos = 0;
        size_t temp = column;
        for (size_t dim = 1; dim < buf_info.ndim; ++dim) {
            size_t index = temp % buf_info.shape[dim];
            start_pos += index * (buf_info.strides[dim] / buf_info.itemsize);
            temp /= buf_info.shape[dim];
        }
        return start_pos;
    }

    // Write each element of a std::tuple<double, double, ...> to consecutive
    // doubles starting at `dest`. Used by the M>1-output dispatcher to
    // serialise a call() result into a contiguous numpy output buffer.
    template <typename Tuple, size_t... Is>
    inline void write_tuple_helper(double* dest, const Tuple& t, std::index_sequence<Is...>) {
        ((dest[Is] = std::get<Is>(t)), ...);
    }

    template <typename Tuple>
    inline void write_tuple_to_memory(double* dest, const Tuple& t) {
        constexpr size_t kSize = std::tuple_size_v<Tuple>;
        write_tuple_helper(dest, t, std::make_index_sequence<kSize>{});
    }

    // Build a 1-D column array from a raw source pointer and pre-divided strides.
    // src        – start of the whole 2-D buffer
    // row_stride – elements between successive rows    (strides[0] / itemsize)
    // col_stride – elements between successive columns (strides[1] / itemsize)
    // j          – column index to extract
    // T          – number of rows
    inline py::array_t<double> extract_column(const double* src,
                                              std::ptrdiff_t row_stride,
                                              std::ptrdiff_t col_stride,
                                              size_t j, size_t T) {
        py::array_t<double> col(static_cast<py::ssize_t>(T));
        double* dst = static_cast<double*>(col.request().ptr);
        for (size_t i = 0; i < T; ++i) {
            dst[i] = src[i * row_stride + j * col_stride];
        }
        return col;
    }

    // If args is a single 2-D (T, N) numpy array, return a tuple of its N
    // columns as 1-D arrays; otherwise return an empty optional. Enforces the
    // exact-width match (shape[1] == N); a mismatched width throws a clear error.
    // The buffer is requested exactly once; column extraction reuses that info.
    template <size_t N>
    inline std::optional<py::tuple> maybe_split_TxN(const py::args& args) {
        if (args.size() != 1 || !py::isinstance<py::array>(args[0])) {
            return std::nullopt;
        }
        py::array_t<double> arr = py::cast<py::array_t<double>>(args[0]);
        py::buffer_info info = arr.request();
        if (info.ndim != 2) {
            return std::nullopt;   // 1-D single array falls through to the normal error
        }
        size_t T = static_cast<size_t>(info.shape[0]);
        size_t width = static_cast<size_t>(info.shape[1]);
        if (width != N) {
            throw py::value_error(
                "This functor expects " + std::to_string(N) +
                " inputs; got a single 2-D array with " + std::to_string(width) +
                " columns. Pass an (T, " + std::to_string(N) + ") array or " +
                std::to_string(N) + " separate arrays.");
        }
        const double* src = static_cast<const double*>(info.ptr);
        std::ptrdiff_t row_stride = info.strides[0] / info.itemsize;
        std::ptrdiff_t col_stride = info.strides[1] / info.itemsize;
        py::tuple cols(N);
        for (size_t j = 0; j < N; ++j) {
            cols[j] = extract_column(src, row_stride, col_stride, j, T);
        }
        return cols;
    }

}

template <class Derived, size_t N, size_t M>
class FunctorBase {
public:
    using InputArray = std::array<double, N>;
    using OutputArray = std::array<double, M>;
    using ResultTuple = std::conditional_t<M == 1, double, typename detail::TupleOfDoubles<M>>;

    // call() is the algorithm: takes N inputs, returns ResultTuple (1 or M doubles).
    // Derived classes override this. Plan B (this session): single-method
    // call interface. The split process_input()/get_output() interface that
    // was being trialled here is reverted until we genuinely need it for
    // sparse-output algorithms (Plan C).
    virtual ResultTuple call(const InputArray& inputs) = 0;
    virtual void reset() {};



    template <size_t TN = N, size_t TM = M, typename = std::enable_if_t<(TN == 1) && (TM == 1)>>
    py::object handle_input_1i_1o_numpy(py::array_t<double>& input) {
        
        py::buffer_info input_info = input.request();

        if (input_info.ndim < 1 || input_info.itemsize != sizeof(double)) {
            throw std::runtime_error("Input array must have at least one dimension and contain doubles");
        }

        // Allocate output storage
        py::array_t<double> output(input_info.shape);
        py::buffer_info output_info = output.request();

        // Get pointers to input and output memory
        double* input_data = static_cast<double*>(input_info.ptr);
        double* output_data = static_cast<double*>(output_info.ptr);

        // get number of elements in a column, and the stepsize
        size_t size = input_info.shape[0];
        std::ptrdiff_t input_stride = input_info.strides[0] / input_info.itemsize;
        std::ptrdiff_t output_stride = output_info.strides[0] / input_info.itemsize;

        // get the number of columns in this ndarray
        auto num_cols = detail::numpy_num_cols(input_info);

        // loop over all columns
        for (size_t col = 0; col < num_cols; ++col) {
            
            // find the start positions in memory of this column
            size_t input_index = detail::numpy_col_start_pos(col, input_info);
            size_t output_index = detail::numpy_col_start_pos(col, output_info);

            reset(); // reset before processing this column

            for (size_t i = 0; i < size; i++) {
                output_data[output_index] = call({input_data[input_index]});

                input_index += input_stride;
                output_index += output_stride;
            }

        }

        reset(); // after we have processed all columns

        return output;
    }

    template <size_t TN = N, size_t TM = M, typename = std::enable_if_t<(TN > 1) && (TM == 1)>>
    py::object handle_input_Ni_1o_numpy(py::tuple& inputs) {

        std::array<py::array_t<double>, TN> inputs_array;
        std::array<py::buffer_info, TN> inputs_info;
        // Check that the first input is a numpy array
        if (!py::isinstance<py::array>(inputs[0])) {
            throw py::type_error("Incompatible input type, a mix of numpy arrays and other.");
        }
        inputs_array[0] = py::cast<py::array_t<double>>(inputs[0]);


        // Get shape info of the first numpy array
        inputs_info[0] = inputs_array[0].request();

        // basic cjeck
        if (inputs_info[0].ndim < 1) {
            throw std::runtime_error("Input array must have at least one dimension");
        }

        // Check alignment between the first numpy array, and all others
        for (size_t i = 1; i < TN; ++i) {
            // Check that that the type is a numpy array
            if (!py::isinstance<py::array>(inputs[i])) {
                throw py::type_error("Incompatible input type, a mix of numpy arrays and other.");
            }
            inputs_array[i] = py::cast<py::array_t<double>>(inputs[i]);

            // Check for the same number of dims
            inputs_info[i] = inputs_array[i].request();
            if (inputs_info[0].ndim != inputs_info[i].ndim) {
                throw py::type_error("Incompatible input numpy arrays, dimensions mismatch.");
            }
            // Check for the same number of elements per dim
            for (size_t d=0; d<inputs_info[0].ndim; ++d) {
                if (inputs_info[0].shape[d] != inputs_info[i].shape[d]) {
                    throw py::type_error("Incompatible input numpy arrays, shape mismatch.");
                }
            }
        }
        // we have N numpy arrays of matching shape!

        // Allocate output storage
        py::array_t<double> output(inputs_info[0].shape);
        py::buffer_info output_info = output.request();
        double* output_data = static_cast<double*>(output_info.ptr);
        std::ptrdiff_t output_stride = output_info.strides[0] / output_info.itemsize;

        // Get input info
        std::array<double*, TN> inputs_data{};
        std::array<int64_t, TN> inputs_stride{};
        size_t size = inputs_info[0].shape[0];
        for (size_t i = 0; i < TN; ++i) {
            inputs_data[i] = static_cast<double*>(inputs_info[i].ptr);
            inputs_stride[i] = inputs_info[i].strides[0] / inputs_info[i].itemsize;
        }

        // get the number of columns in this ndarray
        auto num_cols = detail::numpy_num_cols(inputs_info[0]);

        // loop over all columns
        std::array<size_t, TN> inputs_index{};
        for (size_t col = 0; col < num_cols; ++col) {
            
            // find the start positions in memory of this column for all input arguments
            for (size_t i = 0; i < TN; ++i) {
                inputs_index[i] = detail::numpy_col_start_pos(col, inputs_info[i]);
            }
            size_t output_index = detail::numpy_col_start_pos(col, output_info);

            reset(); // reset before processing this column

            InputArray call_array;
            for (size_t i = 0; i < size; i++) {
                for (size_t i = 0; i < TN; ++i) {
                    call_array[i] = inputs_data[i][inputs_index[i]];
                }
                output_data[output_index] = call(call_array);

                for (size_t i = 0; i < TN; ++i) {
                    inputs_index[i] += inputs_stride[i];
                }
                output_index += output_stride;
            }

        }

        reset(); // after we have processed all columns

        return output;
    }    

    // ---------------------------------------------------------
    // ONE INPUT, M>1 OUTPUTS HANDLER
    // ---------------------------------------------------------
    // For a 1-input array of shape (T, ...) the output has shape
    // (T, ..., M): an extra trailing axis of size M is appended for the
    // M outputs per time step. Memory is written contiguously per step.
    template <size_t TN = N, size_t TM = M, typename = std::enable_if_t<(TN == 1) && (TM > 1)>>
    py::object handle_input_1i_Mo_numpy(py::array_t<double>& input) {
        py::buffer_info input_info = input.request();

        if (input_info.ndim < 1 || input_info.itemsize != sizeof(double)) {
            throw std::runtime_error("Input array must have at least one dimension and contain doubles");
        }

        // Output shape = input shape + (M,)
        std::vector<py::ssize_t> output_shape(input_info.shape.begin(), input_info.shape.end());
        output_shape.push_back(static_cast<py::ssize_t>(M));
        py::array_t<double> output(output_shape);
        py::buffer_info output_info = output.request();

        double* input_data = static_cast<double*>(input_info.ptr);
        double* output_data = static_cast<double*>(output_info.ptr);

        size_t size = input_info.shape[0];
        std::ptrdiff_t input_stride = input_info.strides[0] / input_info.itemsize;
        std::ptrdiff_t output_stride = output_info.strides[0] / output_info.itemsize;

        auto num_cols = detail::numpy_num_cols(input_info);

        for (size_t col = 0; col < num_cols; ++col) {
            size_t input_index = detail::numpy_col_start_pos(col, input_info);
            // The output's spatial dims are the input's followed by an
            // appended size-M axis. For col < num_cols (which is the
            // product of input's spatial dims), numpy_col_start_pos walks
            // the M axis with index 0 because col / product(input.shape[1..])
            // is 0, so it lands at output[0, ..., 0]. The M values are
            // then written at offsets 0..M-1 within that step.
            size_t output_index = detail::numpy_col_start_pos(col, output_info);

            reset();

            for (size_t i = 0; i < size; i++) {
                ResultTuple results = call({input_data[input_index]});
                detail::write_tuple_to_memory(&output_data[output_index], results);

                input_index += input_stride;
                output_index += output_stride;
            }
        }

        reset();
        return output;
    }

    template <size_t TN = N, size_t TM = M, typename = std::enable_if_t<(TN == 1) && (TM > 1)>>
    py::object handle_input_1i_Mo(py::object input) {
        // Case 1: Scalar input -> tuple of M floats
        try {
            InputArray input_array = {input.cast<double>()};
            return py::cast(call(input_array));
        } catch (const py::cast_error&) {
        }

        // Case 2: Numpy array (and lists/tuples cast to arrays at the
        // ScreamerBase boundary; here pure arrays are routed)
        if (py::isinstance<py::array>(input)) {
            py::array_t<double> input_pyarray = py::cast<py::array_t<double>>(input);
            return handle_input_1i_Mo_numpy(input_pyarray);
        }

        // Case 3: Iterable -> list of M-tuples (eager).
        if (py::isinstance<py::iterable>(input)) {
            std::vector<ResultTuple> results;
            for (auto item : input) {
                try {
                    InputArray input_array = {item.cast<double>()};
                    results.push_back(call(input_array));
                } catch (const py::cast_error&) {
                    throw py::type_error("Iterable must contain numbers.");
                }
            }
            return py::cast(results);
        }

        throw py::type_error("Unsupported input type. Supported types are number, numpy array, or iterable.");
    }


    // one input, one output
    template <size_t TN = N, size_t TM = M, typename = std::enable_if_t<(TN == 1) && (TM == 1)>>
    py::object handle_input_1i_1o(py::object input) {
        
        // Case 1: Scalar input
        try {
            InputArray input_array = {input.cast<double>()};
            return py::cast(call(input_array));
        } catch (const py::cast_error&) {
            // If not a scalar, fall through to further checks
        }

        // Case 2: Numpy array
        if (py::isinstance<py::array>(input)) {
            py::array_t<double> input_pyarray = py::cast<py::array_t<double>>(input);
            return handle_input_1i_1o_numpy(input_pyarray);
        }

        // Case 3: Iterable
        if (py::isinstance<py::iterable>(input)) {
            std::vector<ResultTuple> results;

            for (auto item : input) {

                try {
                    // Case 2.1: Scalar input item
                    InputArray input_array = {item.cast<double>()};
                    results.push_back(call(input_array));
                    continue;
                } catch (const py::cast_error&) {
                    // If not a scalar, continue to further checks
                }

                if (py::isinstance<py::tuple>(item)) {
                    auto tuple = item.cast<py::tuple>();
                    if (tuple.size() == N) {
                        InputArray input_array = cast_to_array(tuple);
                        results.push_back(call(input_array));
                    } else {
                        throw py::type_error("Invalid tuple size in iterable.");
                    }
                } else {
                    throw py::type_error("Iterable must contain doubles or tuples of correct size.");
                }
            }

            return py::cast(results);
        }

        // Case no match:
        throw py::type_error("Unsupported input type. Supported types are double, or iterables.");
    }


    py::tuple args_to_tuple_n(const py::args args) const {

        if (args.size() == 1) { // a container of N

            auto arg = args[0];
            
            // we only support tuples and lists
            if (!(py::isinstance<py::list>(arg) || py::isinstance<py::tuple>(arg))) {
                throw py::type_error("Unsupported single argument input type. Supported types are lists or tuples.");
            }

            // convert to tuple
             py::tuple inputs = py::cast<py::tuple>(arg);

            // validate size
            if (inputs.size() != N) {
                throw py::type_error("Wrong number of elements in the single argument input list / tuple.");
            }

            return inputs;
        } 
        
        if (args.size() == N) {
            return py::cast<py::tuple>(args);
        }
        
        throw py::type_error("Wrong number of arguments.");
    }
    

    // ---------------------------------------------------------
    // MULTIPLE INPUTS, ONE OUTPUT HANDELER
    // ---------------------------------------------------------
    template <size_t TN = N, size_t TM = M, typename = std::enable_if_t<(TN > 1) && (TM == 1)>>
    py::object handle_input_Ni_1o(const py::args args) {

        if (auto cols = detail::maybe_split_TxN<N>(args)) {
            return handle_input_Ni_1o_numpy(*cols);
        }

       // Case 1: we need to get his out of the way first.
       // A single argument, list/tuple with N-tuples inside: [ (1,2,3), (4,5,6), ...]
        if (args.size() == 1) {
            auto input = args[0];
            if (py::isinstance<py::list>(input) || py::isinstance<py::tuple>(input)) {
                bool valid = true;
                std::vector<ResultTuple> results;
                for (auto item : input) {
                    if (!py::isinstance<py::tuple>(item)) {
                        valid = false;
                        break;
                    }
                    auto tuple = item.cast<py::tuple>();
                    if (tuple.size() != N) {
                        valid = false;
                        break;
                    }
                    InputArray input_array = cast_to_array(tuple);
                    results.push_back(call(input_array));
                }
                if (valid) {       
                    return py::cast(results);         
                }
            }
        }

        // after this, now we handle cases where we have N arguments
        py::tuple inputs = args_to_tuple_n(args);

        // Case 2: try a tuple of N scalar input: (0.3, 1.2, 4.0)
        try {
            InputArray array;
            for (size_t i = 0; i < N; ++i) {
                array[i] = inputs[i].cast<double>();
            }
            return py::cast(call(array));
        } catch (const py::cast_error&) {
            // If not a scalar, fall through to further checks
        }

        // Case 3: try a tuple of N numpy arrays, all of the same size.( nparray, nparray, nparray)
        if (py::isinstance<py::array>(inputs[0])) {
            return handle_input_Ni_1o_numpy(inputs);
        }

        // Case 4: a tuple of N iterables: ( [... != ], [...], [...] )
        bool all_iterable = true;
        for (auto input : inputs) {
            all_iterable = all_iterable && py::isinstance<py::iterable>(input);
            if (!all_iterable) {
                break;
            }
        }

        if (all_iterable) {

            // Initialize iterators for each input iterable
            std::array<py::iterator, N> iterators;

            for (size_t i = 0; i < N; ++i) {
                iterators[i] = py::iter(inputs[i]);
            }

            std::vector<ResultTuple> results;

            // Loop until any of the iterators is exhausted
            while (true) {
                
                InputArray array;
                try {
                    // Advance all iterators and collect the next items
                    for (size_t i = 0; i < N; ++i) {

                        // Check if the iterator is valid
                        if (iterators[i] == py::iterator()) {
                            throw py::stop_iteration();
                        }

                        auto val = *iterators[i];
                        array[i] = val.template cast<double>();
                        ++iterators[i];
                    }
                } catch (py::stop_iteration&) {
                    // One of the iterators is exhausted; exit the loop
                    break;
                }

                // store the call
                results.push_back(call(array));

            }

            return py::cast(results);

        }

        // Case no match:
        throw py::type_error("Unsupported input type.");
    }

    // ---------------------------------------------------------
    // MULTIPLE INPUTS, MULTIPLE OUTPUTS HANDLER (Plan E)
    // ---------------------------------------------------------
    // The natural composition of the N->1 and 1->M rules:
    //   - inputs are paired column-by-column (from N->1)
    //   - output shape = paired-input shape + (M,) (from 1->M)
    template <size_t TN = N, size_t TM = M, typename = std::enable_if_t<(TN > 1) && (TM > 1)>>
    py::object handle_input_Ni_Mo_numpy(py::tuple& inputs) {

        std::array<py::array_t<double>, TN> inputs_array;
        std::array<py::buffer_info, TN> inputs_info;

        if (!py::isinstance<py::array>(inputs[0])) {
            throw py::type_error("Incompatible input type, a mix of numpy arrays and other.");
        }
        inputs_array[0] = py::cast<py::array_t<double>>(inputs[0]);
        inputs_info[0] = inputs_array[0].request();

        if (inputs_info[0].ndim < 1) {
            throw std::runtime_error("Input array must have at least one dimension");
        }

        for (size_t i = 1; i < TN; ++i) {
            if (!py::isinstance<py::array>(inputs[i])) {
                throw py::type_error("Incompatible input type, a mix of numpy arrays and other.");
            }
            inputs_array[i] = py::cast<py::array_t<double>>(inputs[i]);
            inputs_info[i] = inputs_array[i].request();
            if (inputs_info[0].ndim != inputs_info[i].ndim) {
                throw py::type_error("Incompatible input numpy arrays, dimensions mismatch.");
            }
            for (size_t d = 0; d < inputs_info[0].ndim; ++d) {
                if (inputs_info[0].shape[d] != inputs_info[i].shape[d]) {
                    throw py::type_error("Incompatible input numpy arrays, shape mismatch.");
                }
            }
        }

        // Output shape = paired input shape + (M,)
        std::vector<py::ssize_t> output_shape(inputs_info[0].shape.begin(),
                                              inputs_info[0].shape.end());
        output_shape.push_back(static_cast<py::ssize_t>(M));
        py::array_t<double> output(output_shape);
        py::buffer_info output_info = output.request();
        double* output_data = static_cast<double*>(output_info.ptr);
        std::ptrdiff_t output_stride = output_info.strides[0] / output_info.itemsize;

        std::array<double*, TN> inputs_data{};
        std::array<int64_t, TN> inputs_stride{};
        size_t size = inputs_info[0].shape[0];
        for (size_t i = 0; i < TN; ++i) {
            inputs_data[i] = static_cast<double*>(inputs_info[i].ptr);
            inputs_stride[i] = inputs_info[i].strides[0] / inputs_info[i].itemsize;
        }

        auto num_cols = detail::numpy_num_cols(inputs_info[0]);

        std::array<size_t, TN> inputs_index{};
        for (size_t col = 0; col < num_cols; ++col) {
            for (size_t i = 0; i < TN; ++i) {
                inputs_index[i] = detail::numpy_col_start_pos(col, inputs_info[i]);
            }
            // numpy_col_start_pos walks the trailing M-axis with index 0
            // because col / product(input.shape[1..]) is 0, landing at
            // output[0, ..., 0]. The M values are then written at offsets
            // 0..M-1 within that step (same trick as 1i_Mo).
            size_t output_index = detail::numpy_col_start_pos(col, output_info);

            reset();

            InputArray call_array;
            for (size_t i = 0; i < size; i++) {
                for (size_t j = 0; j < TN; ++j) {
                    call_array[j] = inputs_data[j][inputs_index[j]];
                }
                ResultTuple results = call(call_array);
                detail::write_tuple_to_memory(&output_data[output_index], results);

                for (size_t j = 0; j < TN; ++j) {
                    inputs_index[j] += inputs_stride[j];
                }
                output_index += output_stride;
            }
        }

        reset();
        return output;
    }


    template <size_t TN = N, size_t TM = M, typename = std::enable_if_t<(TN > 1) && (TM > 1)>>
    py::object handle_input_Ni_Mo(const py::args args) {

        if (auto cols = detail::maybe_split_TxN<N>(args)) {
            return handle_input_Ni_Mo_numpy(*cols);
        }

        // Case 1: single argument, list/tuple of N-tuples
        // [(x0, y0), (x1, y1), ...] -> list of M-tuples
        if (args.size() == 1) {
            auto input = args[0];
            if (py::isinstance<py::list>(input) || py::isinstance<py::tuple>(input)) {
                bool valid = true;
                std::vector<ResultTuple> results;
                for (auto item : input) {
                    if (!py::isinstance<py::tuple>(item)) {
                        valid = false;
                        break;
                    }
                    auto tuple = item.cast<py::tuple>();
                    if (tuple.size() != N) {
                        valid = false;
                        break;
                    }
                    InputArray input_array = cast_to_array(tuple);
                    results.push_back(call(input_array));
                }
                if (valid) {
                    return py::cast(results);
                }
            }
        }

        py::tuple inputs = args_to_tuple_n(args);

        // Case 2: tuple of N scalars -> single M-tuple
        try {
            InputArray array;
            for (size_t i = 0; i < N; ++i) {
                array[i] = inputs[i].cast<double>();
            }
            return py::cast(call(array));
        } catch (const py::cast_error&) {
        }

        // Case 3: tuple of N numpy arrays of matching shape
        if (py::isinstance<py::array>(inputs[0])) {
            return handle_input_Ni_Mo_numpy(inputs);
        }

        // Case 4: tuple of N iterables -> list of M-tuples (eager)
        bool all_iterable = true;
        for (auto input : inputs) {
            all_iterable = all_iterable && py::isinstance<py::iterable>(input);
            if (!all_iterable) {
                break;
            }
        }

        if (all_iterable) {
            std::array<py::iterator, N> iterators;
            for (size_t i = 0; i < N; ++i) {
                iterators[i] = py::iter(inputs[i]);
            }

            std::vector<ResultTuple> results;
            while (true) {
                InputArray array;
                try {
                    for (size_t i = 0; i < N; ++i) {
                        if (iterators[i] == py::iterator()) {
                            throw py::stop_iteration();
                        }
                        auto val = *iterators[i];
                        array[i] = val.template cast<double>();
                        ++iterators[i];
                    }
                } catch (py::stop_iteration&) {
                    break;
                }
                results.push_back(call(array));
            }

            return py::cast(results);
        }

        throw py::type_error("Unsupported input type.");
    }


    // ---------------------------------------------------------
    // Main dispatcher
    // ---------------------------------------------------------
    py::object handle_input(py::args args) {
        for (auto a : args) {
            if (screamer::is_dag_node(py::reinterpret_borrow<py::object>(a))) {
                py::object self = py::cast(static_cast<Derived*>(this));
                return screamer::make_dag_functor_node(self, py::cast<py::tuple>(args));
            }
        }
        if constexpr (N == 1) {
            if (args.size() != 1) {
                throw py::type_error("Wrong number of in puts");
            }
        }
        if constexpr ((N == 1) && (M == 1)) {
            return handle_input_1i_1o(args[0]);
        } else if constexpr ((N > 1) && (M == 1)) {
            return handle_input_Ni_1o(args);
        } else if constexpr ((N == 1) && (M > 1)) {
            return handle_input_1i_Mo(args[0]);
        } else if constexpr ((N > 1) && (M > 1)) {
            return handle_input_Ni_Mo(args);
        } else {
            throw py::type_error("Unknown configuration.");
        }
    }

    

private:
    InputArray cast_to_array(const py::tuple& tuple) {
        if (tuple.size() != N) {
            throw py::type_error("Tuple size does not match the number of expected inputs.");
        }
        InputArray array;
        for (size_t i = 0; i < N; ++i) {
            array[i] = tuple[i].cast<double>();
        }
        return array;
    }
};




}
#endif