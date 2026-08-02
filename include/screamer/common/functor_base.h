#ifndef FUNCTOR_BASE_H
#define FUNCTOR_BASE_H

#include <array>
#include <cstddef>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>
#include <cstdint>
#include "screamer/common/eval_op.h"

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

}

template <class Derived, size_t N, size_t M>
class FunctorBase : public EvalOp {
public:
    static constexpr size_t kN = N;
    static constexpr size_t kM = M;
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
};

}
#endif
