#ifndef FUNCTOR_BASE_H
#define FUNCTOR_BASE_H

#include <array>
#include <cstddef>
#include <optional>
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/tuple.h>
#include "screamer/common/base.h"
#include "screamer/common/eval_op.h"
#include "screamer/common/lazy_eval_iterator.h"
#include <cstdint>

namespace nb = nanobind;

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

    // Read N numpy arrays (any dtype/strides) into contiguous double columns,
    // validating that all N share the same shape. Mirrors the pybind
    // forcecast-and-check path. Throws the same error vocabulary the tests
    // assert (TypeError on a mix / shape mismatch).
    template <size_t N>
    inline std::array<ContigDouble, N> read_n_arrays(const nb::tuple& inputs) {
        std::array<ContigDouble, N> out;
        if (!is_ndarray(inputs[0])) {
            throw nb::type_error("Incompatible input type, a mix of numpy arrays and other.");
        }
        out[0] = read_contig_double(nb::cast<nb::ndarray<>>(inputs[0]));
        if (out[0].ndim() < 1) {
            throw std::runtime_error("Input array must have at least one dimension");
        }
        for (size_t i = 1; i < N; ++i) {
            if (!is_ndarray(inputs[i])) {
                throw nb::type_error("Incompatible input type, a mix of numpy arrays and other.");
            }
            out[i] = read_contig_double(nb::cast<nb::ndarray<>>(inputs[i]));
            if (out[i].ndim() != out[0].ndim()) {
                throw nb::type_error("Incompatible input numpy arrays, dimensions mismatch.");
            }
            for (size_t d = 0; d < out[0].ndim(); ++d) {
                if (out[0].shape[d] != out[i].shape[d]) {
                    throw nb::type_error("Incompatible input numpy arrays, shape mismatch.");
                }
            }
        }
        return out;
    }

    // If args is a single 2-D (T, N) numpy array, return its N columns as
    // contiguous 1-D double arrays; otherwise an empty optional. Enforces the
    // exact-width match (shape[1] == N); a mismatched width throws a clear error.
    template <size_t N>
    inline std::optional<std::array<ContigDouble, N>> maybe_split_TxN(const nb::args& args) {
        if (args.size() != 1 || !is_ndarray(args[0])) {
            return std::nullopt;
        }
        nb::ndarray<> arr = nb::cast<nb::ndarray<>>(args[0]);
        if (arr.ndim() != 2) {
            return std::nullopt;   // 1-D single array falls through to the normal error
        }
        size_t T = arr.shape(0);
        size_t width = arr.shape(1);
        if (width != N) {
            throw nb::value_error(
                ("This functor expects " + std::to_string(N) +
                 " inputs; got a single 2-D array with " + std::to_string(width) +
                 " columns. Pass an (T, " + std::to_string(N) + ") array or " +
                 std::to_string(N) + " separate arrays.").c_str());
        }
        ContigDouble full = read_contig_double(arr);   // C-contiguous (T, N)
        std::array<ContigDouble, N> cols;
        for (size_t j = 0; j < N; ++j) {
            cols[j].shape = { T };
            cols[j].data.resize(T);
            for (size_t i = 0; i < T; ++i) cols[j].data[i] = full.data[i * N + j];
        }
        return cols;
    }

}

template <class Derived, size_t N, size_t M>
class FunctorBase : public EvalOp {
public:
    using InputArray = std::array<double, N>;
    using OutputArray = std::array<double, M>;
    using ResultTuple = std::conditional_t<M == 1, double, typename detail::TupleOfDoubles<M>>;

    // Optional batch path, for operators whose whole-array algorithm differs
    // from replaying the per-event one. Return false to decline, which is the
    // default and leaves the per-sample loop over call() in charge.
    //
    // The sliding-window extremum is the case this exists for: over an array
    // it can be computed by block decomposition, with no data-dependent
    // branching, but that needs lookahead and so cannot be the event path.
    // See detail/block_extremum.h.
    //
    // The operator is freshly reset when this is called, and must leave state
    // as the per-sample loop would. Overrides are expected to decline when
    // they cannot apply (non-unit strides, NaN in the input) rather than
    // change any result.
    virtual bool process_columns(
        double* /*output*/, std::ptrdiff_t /*output_stride*/,
        const std::array<double*, N>& /*inputs*/,
        const std::array<int64_t, N>& /*input_strides*/,
        const std::array<size_t, N>& /*input_offsets*/,
        size_t /*output_offset*/, size_t /*size*/) { return false; }

    // call() is the algorithm: takes N inputs, returns ResultTuple (1 or M doubles).
    // Derived classes override this single method. There is no separate
    // process_input()/get_output() split; a sparse-output variant would need one.
    virtual ResultTuple call(const InputArray& inputs) = 0;
    void reset() override {}

    // EvalOp interface: uniform N-in / M-out entry point for the DAG engine.
    std::size_t n_in() const override { return N; }
    std::size_t n_out() const override { return M; }
    void eval(const double* in, double* out) override {
        InputArray inputs;
        for (std::size_t i = 0; i < N; ++i) inputs[i] = in[i];
        if constexpr (M == 1) {
            out[0] = call(inputs);
        } else {
            detail::write_tuple_to_memory(out, call(inputs));
        }
    }


    // ---------------------------------------------------------
    // ONE INPUT, ONE OUTPUT (numpy)
    // ---------------------------------------------------------
    template <size_t TN = N, size_t TM = M, typename = std::enable_if_t<(TN == 1) && (TM == 1)>>
    nb::object handle_input_1i_1o_numpy(const nb::ndarray<>& input) {
        detail::ContigDouble in = detail::read_contig_double(input);
        if (in.ndim() < 1) {
            throw std::runtime_error("Input array must have at least one dimension and contain doubles");
        }
        std::vector<size_t> shape = in.shape;
        size_t total = in.total();
        double* out = new double[total ? total : 1];
        size_t size = in.time();
        size_t cols = in.cols();   // product of trailing dims = C-contiguous axis0 stride

        for (size_t col = 0; col < cols; ++col) {
            reset();
            size_t idx = col;
            for (size_t i = 0; i < size; ++i) {
                out[idx] = call({ in.data[idx] });
                idx += cols;
            }
        }
        reset();
        return detail::make_owned_array(out, shape);
    }

    // ---------------------------------------------------------
    // MULTIPLE INPUTS, ONE OUTPUT (numpy)
    // ---------------------------------------------------------
    template <size_t TN = N, size_t TM = M, typename = std::enable_if_t<(TN > 1) && (TM == 1)>>
    nb::object handle_input_Ni_1o_numpy(std::array<detail::ContigDouble, N>& ins) {
        if (ins[0].ndim() < 1) {
            throw std::runtime_error("Input array must have at least one dimension");
        }
        std::vector<size_t> shape = ins[0].shape;
        size_t total = ins[0].total();
        size_t size = ins[0].time();
        size_t cols = ins[0].cols();
        double* out = new double[total ? total : 1];

        std::array<double*, N> inptr{};
        std::array<int64_t, N> instride{};
        for (size_t i = 0; i < N; ++i) { inptr[i] = ins[i].data.data(); instride[i] = (int64_t) cols; }

        for (size_t col = 0; col < cols; ++col) {
            std::array<size_t, N> inoff{};
            for (size_t i = 0; i < N; ++i) inoff[i] = col;

            reset();

            if (!process_columns(out, (std::ptrdiff_t) cols, inptr, instride, inoff, col, size)) {
                std::array<size_t, N> idx_in = inoff;
                size_t idx_out = col;
                InputArray call_array;
                for (size_t t = 0; t < size; ++t) {
                    for (size_t i = 0; i < N; ++i) call_array[i] = inptr[i][idx_in[i]];
                    out[idx_out] = call(call_array);
                    for (size_t i = 0; i < N; ++i) idx_in[i] += cols;
                    idx_out += cols;
                }
            }
        }
        reset();
        return detail::make_owned_array(out, shape);
    }

    // ---------------------------------------------------------
    // ONE INPUT, M>1 OUTPUTS (numpy)
    // ---------------------------------------------------------
    // For a 1-input array of shape (T, ...) the output has shape
    // (T, ..., M): an extra trailing axis of size M is appended for the
    // M outputs per time step. Memory is written contiguously per step.
    template <size_t TN = N, size_t TM = M, typename = std::enable_if_t<(TN == 1) && (TM > 1)>>
    nb::object handle_input_1i_Mo_numpy(const nb::ndarray<>& input) {
        detail::ContigDouble in = detail::read_contig_double(input);
        if (in.ndim() < 1) {
            throw std::runtime_error("Input array must have at least one dimension and contain doubles");
        }
        std::vector<size_t> shape = in.shape;
        shape.push_back(M);
        size_t total = in.total();
        double* out = new double[total * M ? total * M : 1];
        size_t size = in.time();
        size_t cols = in.cols();

        for (size_t col = 0; col < cols; ++col) {
            reset();
            for (size_t i = 0; i < size; ++i) {
                size_t in_idx = i * cols + col;
                ResultTuple results = call({ in.data[in_idx] });
                detail::write_tuple_to_memory(&out[in_idx * M], results);
            }
        }
        reset();
        return detail::make_owned_array(out, shape);
    }

    // ---------------------------------------------------------
    // MULTIPLE INPUTS, MULTIPLE OUTPUTS (numpy)
    // ---------------------------------------------------------
    // The natural composition of the N->1 and 1->M rules:
    //   - inputs are paired column-by-column (from N->1)
    //   - output shape = paired-input shape + (M,) (from 1->M)
    template <size_t TN = N, size_t TM = M, typename = std::enable_if_t<(TN > 1) && (TM > 1)>>
    nb::object handle_input_Ni_Mo_numpy(std::array<detail::ContigDouble, N>& ins) {
        if (ins[0].ndim() < 1) {
            throw std::runtime_error("Input array must have at least one dimension");
        }
        std::vector<size_t> shape = ins[0].shape;
        shape.push_back(M);
        size_t total = ins[0].total();
        size_t size = ins[0].time();
        size_t cols = ins[0].cols();
        double* out = new double[total * M ? total * M : 1];

        std::array<double*, N> inptr{};
        for (size_t i = 0; i < N; ++i) inptr[i] = ins[i].data.data();

        for (size_t col = 0; col < cols; ++col) {
            reset();
            InputArray call_array;
            for (size_t t = 0; t < size; ++t) {
                size_t in_idx = t * cols + col;
                for (size_t j = 0; j < N; ++j) call_array[j] = inptr[j][in_idx];
                ResultTuple results = call(call_array);
                detail::write_tuple_to_memory(&out[in_idx * M], results);
            }
        }
        reset();
        return detail::make_owned_array(out, shape);
    }


    // ---------------------------------------------------------
    // ONE INPUT, M>1 OUTPUTS
    // ---------------------------------------------------------
    template <size_t TN = N, size_t TM = M, typename = std::enable_if_t<(TN == 1) && (TM > 1)>>
    nb::object handle_input_1i_Mo(nb::object input) {
        // Case 1: Numpy array. Checked before the scalar cast so a length-1 array
        // is a time series of one (array in, array out - Rule A), not a scalar.
        if (is_series_array(input)) {
            return handle_input_1i_Mo_numpy(nb::cast<nb::ndarray<>>(input));
        }

        // Case 2: Scalar input -> tuple of M floats (one streaming event)
        {
            double d;
            if (nb::try_cast<double>(input, d)) {
                InputArray input_array = { d };
                return nb::cast(call(input_array));
            }
        }

        // Case 3: Iterable.
        // True lazy iterators (generators, iter(...)) stream via LazyEvalIterator.
        // Concrete list/tuple inputs are eager and return a list.
        if (nb::hasattr(input, "__iter__")) {
            if (is_lazy_iterable(input)) {
                std::vector<nb::object> sources{ nb::borrow<nb::object>(input) };
                return nb::cast(LazyEvalIterator(nb::find(*static_cast<Derived*>(this)), std::move(sources)));
            }
            // Eager path for list/tuple input.
            std::vector<ResultTuple> results;
            for (nb::handle item : input) {
                double d;
                if (!nb::try_cast<double>(nb::borrow<nb::object>(item), d)) {
                    throw nb::type_error("Iterable must contain numbers.");
                }
                InputArray input_array = { d };
                results.push_back(call(input_array));
            }
            return nb::cast(results);
        }

        throw nb::type_error("Unsupported input type. Supported types are number, numpy array, or iterable.");
    }


    // ---------------------------------------------------------
    // ONE INPUT, ONE OUTPUT
    // ---------------------------------------------------------
    template <size_t TN = N, size_t TM = M, typename = std::enable_if_t<(TN == 1) && (TM == 1)>>
    nb::object handle_input_1i_1o(nb::object input) {

        // Case 1: Numpy array. Checked before the scalar cast so a length-1 array
        // is a time series of one (array in, array out - Rule A), not a scalar;
        // only an actual Python scalar returns a scalar.
        if (is_series_array(input)) {
            return handle_input_1i_1o_numpy(nb::cast<nb::ndarray<>>(input));
        }

        // Case 2: Scalar input (one streaming event)
        {
            double d;
            if (nb::try_cast<double>(input, d)) {
                InputArray input_array = { d };
                return nb::cast(call(input_array));
            }
        }

        // Case 3: Iterable
        if (nb::hasattr(input, "__iter__")) {
            std::vector<ResultTuple> results;

            for (nb::handle item_h : input) {
                nb::object item = nb::borrow<nb::object>(item_h);

                double d;
                if (nb::try_cast<double>(item, d)) {
                    // Case 2.1: Scalar input item
                    InputArray input_array = { d };
                    results.push_back(call(input_array));
                    continue;
                }

                if (nb::isinstance<nb::tuple>(item)) {
                    nb::tuple tuple = nb::cast<nb::tuple>(item);
                    if (tuple.size() == N) {
                        InputArray input_array = cast_to_array(tuple);
                        results.push_back(call(input_array));
                    } else {
                        throw nb::type_error("Invalid tuple size in iterable.");
                    }
                } else {
                    throw nb::type_error("Iterable must contain doubles or tuples of correct size.");
                }
            }

            return nb::cast(results);
        }

        // Case no match:
        throw nb::type_error("Unsupported input type. Supported types are double, or iterables.");
    }


    nb::tuple args_to_tuple_n(const nb::args& args) const {

        if (args.size() == 1) { // a container of N

            nb::object arg = nb::borrow<nb::object>(args[0]);

            // we only support tuples and lists
            if (!(nb::isinstance<nb::list>(arg) || nb::isinstance<nb::tuple>(arg))) {
                throw nb::type_error("Unsupported single argument input type. Supported types are lists or tuples.");
            }

            // convert to tuple
            nb::tuple inputs = nb::cast<nb::tuple>(arg);

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


    // ---------------------------------------------------------
    // MULTIPLE INPUTS, ONE OUTPUT
    // ---------------------------------------------------------
    template <size_t TN = N, size_t TM = M, typename = std::enable_if_t<(TN > 1) && (TM == 1)>>
    nb::object handle_input_Ni_1o(const nb::args& args) {

        if (auto cols = detail::maybe_split_TxN<N>(args)) {
            return handle_input_Ni_1o_numpy(*cols);
        }

       // Case 1: we need to get his out of the way first.
       // A single argument, list/tuple with N-tuples inside: [ (1,2,3), (4,5,6), ...]
        if (args.size() == 1) {
            nb::object input = nb::borrow<nb::object>(args[0]);
            if (nb::isinstance<nb::list>(input) || nb::isinstance<nb::tuple>(input)) {
                bool valid = true;
                std::vector<ResultTuple> results;
                for (nb::handle item : input) {
                    if (!nb::isinstance<nb::tuple>(item)) {
                        valid = false;
                        break;
                    }
                    nb::tuple tuple = nb::cast<nb::tuple>(item);
                    if (tuple.size() != N) {
                        valid = false;
                        break;
                    }
                    InputArray input_array = cast_to_array(tuple);
                    results.push_back(call(input_array));
                }
                if (valid) {
                    return nb::cast(results);
                }
            }
        }

        // after this, now we handle cases where we have N arguments
        nb::tuple inputs = args_to_tuple_n(args);

        // Case 2: a tuple of N numpy arrays, all of the same size (nparray, ...).
        // Checked before the scalar cast so N length-1 arrays are a series of one
        // (array in, array out - Rule A), not N scalars collapsing to one scalar.
        if (is_series_array(inputs[0])) {
            auto cols = detail::read_n_arrays<N>(inputs);
            return handle_input_Ni_1o_numpy(cols);
        }

        // Case 3: a tuple of N scalar inputs: (0.3, 1.2, 4.0) -> one scalar event
        {
            InputArray array;
            bool ok = true;
            for (size_t i = 0; i < N; ++i) {
                if (!nb::try_cast<double>(nb::borrow<nb::object>(inputs[i]), array[i])) { ok = false; break; }
            }
            if (ok) return nb::cast(call(array));
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
                return nb::cast(LazyEvalIterator(nb::find(*static_cast<Derived*>(this)), std::move(sources)));
            }
            // Eager path: iterate over each input in parallel.
            return eager_parallel(inputs);
        }

        // Case no match:
        throw nb::type_error("Unsupported input type.");
    }

    // ---------------------------------------------------------
    // MULTIPLE INPUTS, MULTIPLE OUTPUTS
    // ---------------------------------------------------------
    template <size_t TN = N, size_t TM = M, typename = std::enable_if_t<(TN > 1) && (TM > 1)>>
    nb::object handle_input_Ni_Mo(const nb::args& args) {

        if (auto cols = detail::maybe_split_TxN<N>(args)) {
            return handle_input_Ni_Mo_numpy(*cols);
        }

        // Case 1: single argument, list/tuple of N-tuples
        // [(x0, y0), (x1, y1), ...] -> list of M-tuples
        if (args.size() == 1) {
            nb::object input = nb::borrow<nb::object>(args[0]);
            if (nb::isinstance<nb::list>(input) || nb::isinstance<nb::tuple>(input)) {
                bool valid = true;
                std::vector<ResultTuple> results;
                for (nb::handle item : input) {
                    if (!nb::isinstance<nb::tuple>(item)) {
                        valid = false;
                        break;
                    }
                    nb::tuple tuple = nb::cast<nb::tuple>(item);
                    if (tuple.size() != N) {
                        valid = false;
                        break;
                    }
                    InputArray input_array = cast_to_array(tuple);
                    results.push_back(call(input_array));
                }
                if (valid) {
                    return nb::cast(results);
                }
            }
        }

        nb::tuple inputs = args_to_tuple_n(args);

        // Case 2: tuple of N numpy arrays of matching shape.
        if (is_series_array(inputs[0])) {
            auto cols = detail::read_n_arrays<N>(inputs);
            return handle_input_Ni_Mo_numpy(cols);
        }

        // Case 3: tuple of N scalars -> single M-tuple (one streaming event)
        {
            InputArray array;
            bool ok = true;
            for (size_t i = 0; i < N; ++i) {
                if (!nb::try_cast<double>(nb::borrow<nb::object>(inputs[i]), array[i])) { ok = false; break; }
            }
            if (ok) return nb::cast(call(array));
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
                return nb::cast(LazyEvalIterator(nb::find(*static_cast<Derived*>(this)), std::move(sources)));
            }
            // Eager path: iterate over each input in parallel.
            return eager_parallel(inputs);
        }

        throw nb::type_error("Unsupported input type.");
    }


    // ---------------------------------------------------------
    // Main dispatcher
    // ---------------------------------------------------------
    nb::object handle_input(nb::args args) {
        for (nb::handle a : args) {
            if (screamer::is_dag_node(nb::borrow<nb::object>(a))) {
                nb::object self = nb::find(*static_cast<Derived*>(this));
                return screamer::make_dag_functor_node(self, nb::cast<nb::tuple>(args));
            }
        }
        if constexpr (N == 1) {
            if (args.size() != 1) {
                throw nb::type_error("Wrong number of in puts");
            }
        }
        if constexpr ((N == 1) && (M == 1)) {
            return handle_input_1i_1o(nb::borrow<nb::object>(args[0]));
        } else if constexpr ((N > 1) && (M == 1)) {
            return handle_input_Ni_1o(args);
        } else if constexpr ((N == 1) && (M > 1)) {
            return handle_input_1i_Mo(nb::borrow<nb::object>(args[0]));
        } else if constexpr ((N > 1) && (M > 1)) {
            return handle_input_Ni_Mo(args);
        } else {
            throw nb::type_error("Unknown configuration.");
        }
    }



private:
    // Returns true iff h is a numpy array of rank >= 1 (a time series). A 0-d
    // array is rank 0 (one sample, no time axis) and behaves like a scalar, so it
    // is excluded here and falls through to the scalar cast.
    static bool is_series_array(nb::handle h) {
        return detail::is_ndarray(h) && nb::cast<int>(h.attr("ndim")) > 0;
    }

    // Returns true iff h is an iterable that is NOT a list, tuple, or array.
    // Generators and iter(...) objects satisfy this; raw list/tuple do not.
    static bool is_lazy_iterable(nb::handle h) {
        return nb::hasattr(h, "__iter__")
            && !nb::isinstance<nb::list>(h)
            && !nb::isinstance<nb::tuple>(h)
            && !detail::is_ndarray(h);
    }

    // Eager parallel iteration over N iterables: pull one item from each per
    // step, stop when any is exhausted. Returns a list of ResultTuple.
    nb::object eager_parallel(const nb::tuple& inputs) {
        std::array<nb::object, N> iters;
        for (size_t i = 0; i < N; ++i) iters[i] = nb::borrow<nb::object>(inputs[i]).attr("__iter__")();
        std::vector<ResultTuple> results;
        while (true) {
            InputArray array;
            bool stop = false;
            for (size_t i = 0; i < N; ++i) {
                nb::object item;
                try {
                    item = iters[i].attr("__next__")();
                } catch (nb::python_error& e) {
                    if (e.matches(PyExc_StopIteration)) { stop = true; break; }
                    throw;
                }
                array[i] = nb::cast<double>(item);
            }
            if (stop) break;
            results.push_back(call(array));
        }
        return nb::cast(results);
    }

    InputArray cast_to_array(const nb::tuple& tuple) {
        if (tuple.size() != N) {
            throw nb::type_error("Tuple size does not match the number of expected inputs.");
        }
        InputArray array;
        for (size_t i = 0; i < N; ++i) {
            array[i] = nb::cast<double>(tuple[i]);
        }
        return array;
    }
};




}
#endif
