#ifndef FUNCTOR_BASE_H
#define FUNCTOR_BASE_H

#include <algorithm>
#include <array>
#include <cstddef>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>
#include <cstdint>
#include "screamer/common/eval_op.h"

// Inline the operator's call() into the batch column loop. Clang's inliner
// otherwise keeps larger calls out of line (always under the WASM build's -Oz):
// RollingCorr runs 1.7x faster in WASM and 1.35x faster natively (Apple M,
// clang) with it. GCC and MSVC builds keep their compilers' own inlining.
// eval_column itself stays out of line, so each operator's column loop is
// compiled once, the same for every caller. The statement attribute crashes
// older clang (Apple clang 15, Xcode 15); it is used from Apple clang 17 and
// LLVM 18 on, and older compilers go without it.
#if defined(__clang__) && ((defined(__apple_build_version__) && __clang_major__ >= 17) || \
                         (!defined(__apple_build_version__) && __clang_major__ >= 18))
#define SCREAMER_INLINE_CALL [[clang::always_inline]]
#define SCREAMER_NOINLINE_CALL [[clang::noinline]]
#else
#define SCREAMER_INLINE_CALL
#define SCREAMER_NOINLINE_CALL
#endif
#if defined(__GNUC__) || defined(__clang__)
#define SCREAMER_NOINLINE __attribute__((noinline))
#elif defined(_MSC_VER)
#define SCREAMER_NOINLINE __declspec(noinline)
#else
#define SCREAMER_NOINLINE
#endif

namespace screamer {

namespace detail {
    // Derived::kInlineColumnCall, or true when the operator does not set it.
    // An operator sets it to false when its step measures faster as a separate
    // call than inlined into the column loop; none does at present (the
    // two-input EW operators and ADOSC did, and now use kColumnCopy). Its long
    // strided columns then run in place, not tiled (tight_column,
    // column_driver.h).
    template <class D, class = void>
    struct inline_column_call : std::true_type {};
    template <class D>
    struct inline_column_call<D, std::void_t<decltype(D::kInlineColumnCall)>>
        : std::bool_constant<D::kInlineColumnCall> {};

    // Derived::kColumnCopy, or false. An operator whose state is a few plain
    // values (no heap buffers) sets it to true: the column loop then runs on
    // a local copy of the operator and copies it back once, so the compiler
    // keeps the state in registers instead of storing and reloading it every
    // event (the output pointer could alias the operator's members). The
    // speed of that per-event store and reload depends on the process, so
    // the copy also makes the loop's speed predictable.
    template <class D, class = void>
    struct column_copy : std::false_type {};
    template <class D>
    struct column_copy<D, std::void_t<decltype(D::kColumnCopy)>>
        : std::bool_constant<D::kColumnCopy> {};

    // Derived::Stepper, or none. The same idea as kColumnCopy for an operator
    // whose state includes heap buffers (a window), where copying the whole
    // operator would not keep the state in registers. A Stepper is a small
    // value: the plain running state, raw pointers to the buffers, and the
    // constants, with call(inputs) for one event. Derived::stepper() returns
    // one for the current state and Derived::store(stepper) writes its state
    // back. The column loop runs a local Stepper; Derived::call runs the same
    // Stepper::call for one event, so both paths compute alike.
    template <class D, class = void>
    struct has_stepper : std::false_type {};
    template <class D>
    struct has_stepper<D, std::void_t<typename D::Stepper>> : std::true_type {};


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

    bool tight_column() const override { return detail::inline_column_call<Derived>::value; }

    // The per-sample loop over call(), once for every operator. The calls are
    // qualified with Derived, so they bind statically and can be inlined: the
    // CRTP contract is that Derived is the most-derived class that changes
    // the computation. A subclass that only renames an operator inherits this
    // loop; one that changes it, such as detail::SwapTwoInputs
    // (microstructure_synonyms.h), overrides call() and eval_column(). An
    // operator that overrides
    // process_columns gets it first, as a whole-column batch path.
    SCREAMER_NOINLINE
    void eval_column(const double* const* in, const std::ptrdiff_t* in_stride,
                     std::size_t T, double* out, std::ptrdiff_t out_stride) override {
        Derived& self = static_cast<Derived&>(*this);
        self.Derived::reset();
        if constexpr (M == 1 && !std::is_same_v<decltype(&Derived::process_columns),
                                                 decltype(&FunctorBase::process_columns)>) {
            std::array<double*, N> cols;
            std::array<int64_t, N> strides;
            for (std::size_t i = 0; i < N; ++i) {
                cols[i] = const_cast<double*>(in[i]);
                strides[i] = static_cast<int64_t>(in_stride[i]);
            }
            if (self.Derived::process_columns(out, out_stride, cols, strides,
                                              std::array<size_t, N>{}, 0, T))
                return;
        }
        run_column(in, in_stride, T, out, out_stride, std::make_index_sequence<N>{});
    }

private:
    // The loop of eval_column. The input reads are a fold over the N inputs,
    // unrolled at compile time, so the pointers and strides stay in registers
    // at every optimization level.
    template <std::size_t... I>
    void run_column(const double* const* in, const std::ptrdiff_t* in_stride, std::size_t T,
                    double* out, std::ptrdiff_t out_stride, std::index_sequence<I...>) {
        Derived& self = static_cast<Derived&>(*this);
        std::array<const double*, N> p{ in[I]... };
        const std::array<std::ptrdiff_t, N> s{ in_stride[I]... };
        if constexpr (detail::has_stepper<Derived>::value) {
            typename Derived::Stepper k = self.Derived::stepper();
            for (std::size_t t = 0; t < T; ++t, out += out_stride) {
                const InputArray inputs{ *p[I]... };
                ((p[I] += s[I]), ...);
                if constexpr (M == 1) {
                    SCREAMER_INLINE_CALL *out = k.call(inputs);
                } else {
                    SCREAMER_INLINE_CALL detail::write_tuple_to_memory(out, k.call(inputs));
                }
            }
            self.Derived::store(k);
            return;
        }
        if constexpr (detail::column_copy<Derived>::value) {
            Derived local(self);
            for (std::size_t t = 0; t < T; ++t, out += out_stride) {
                const InputArray inputs{ *p[I]... };
                ((p[I] += s[I]), ...);
                if constexpr (M == 1) {
                    SCREAMER_INLINE_CALL *out = local.Derived::call(inputs);
                } else {
                    SCREAMER_INLINE_CALL detail::write_tuple_to_memory(
                        out, local.Derived::call(inputs));
                }
            }
            self = local;
            return;
        }
        for (std::size_t t = 0; t < T; ++t, out += out_stride) {
            const InputArray inputs{ *p[I]... };
            ((p[I] += s[I]), ...);
            if constexpr (!detail::inline_column_call<Derived>::value) {
                if constexpr (M == 1) {
                    SCREAMER_NOINLINE_CALL *out = self.Derived::call(inputs);
                } else {
                    SCREAMER_NOINLINE_CALL detail::write_tuple_to_memory(out, self.Derived::call(inputs));
                }
            } else if constexpr (M == 1) {
                SCREAMER_INLINE_CALL *out = self.Derived::call(inputs);
            } else {
                SCREAMER_INLINE_CALL detail::write_tuple_to_memory(out, self.Derived::call(inputs));
            }
        }
    }
};

// The base of an operator with a Stepper (detail::has_stepper): call() runs
// Stepper::call for one event on a Stepper of the stored state and stores the
// state back, the same step the batch column loop runs.
template <class Derived, size_t N, size_t M>
class StepperFunctorBase : public FunctorBase<Derived, N, M> {
public:
    using typename FunctorBase<Derived, N, M>::InputArray;
    using typename FunctorBase<Derived, N, M>::ResultTuple;

    ResultTuple call(const InputArray& inputs) override {
        Derived& self = static_cast<Derived&>(*this);
        typename Derived::Stepper k = self.Derived::stepper();
        ResultTuple r;
        SCREAMER_INLINE_CALL r = k.call(inputs);
        self.Derived::store(k);
        return r;
    }
};

}
#endif
