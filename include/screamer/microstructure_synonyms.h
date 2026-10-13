#ifndef SCREAMER_MICROSTRUCTURE_SYNONYMS_H
#define SCREAMER_MICROSTRUCTURE_SYNONYMS_H

// Microstructure operators that reuse another operator's kernel under a
// market-microstructure name and input names:
//
//     SignedVolume(sign, volume)                 = Mul(sign, volume)
//     QueueImbalance(bid_size, ask_size)         = OFI(bid_size, ask_size)
//     RollingOrderImbalance(signed_flow)         = RollingSum(signed_flow)
//     RollingKyleLambda(signed_flow, return_)    = RollingBeta(return_, signed_flow)
//     EwKyleLambda(signed_flow, return_)         = EwBeta(return_, signed_flow)
//
// Kyle's lambda is the slope of return on signed flow, so the two Kyle
// operators swap their inputs before the beta kernel sees them. Each class
// lists its inputs in kInputNames, which the Python binding uses to accept
// them as keywords.

#include <array>
#include <cstddef>
#include "screamer/arithmetic.h"
#include "screamer/ofi.h"
#include "screamer/rolling_sum.h"
#include "screamer/rolling_beta.h"
#include "screamer/ew_beta.h"

namespace screamer {

namespace detail {

// A two-input operator that passes its inputs to Base in the reverse order.
// Every path into the kernel is one of the two overrides: call() for one
// event (and EvalOp::eval, which calls it) and eval_column() for a batch
// column. The column swaps its two input pointers and strides, so the batch
// loop runs Base's own inlined kernel with no copy.
template <class Base>
class SwapTwoInputs : public Base {
public:
    static_assert(Base::kN == 2, "SwapTwoInputs needs a two-input operator");
    using Base::Base;
    using typename Base::InputArray;
    using typename Base::ResultTuple;

    ResultTuple call(const InputArray& in) override {
        return Base::call(InputArray{ in[1], in[0] });
    }

    void eval_column(const double* const* in, const std::ptrdiff_t* in_stride,
                     std::size_t T, double* out, std::ptrdiff_t out_stride) override {
        const double* const swapped[2] = { in[1], in[0] };
        const std::ptrdiff_t strides[2] = { in_stride[1], in_stride[0] };
        Base::eval_column(swapped, strides, T, out, out_stride);
    }
};

}  // namespace detail

// Signed order flow, sign * volume. The Mul kernel.
class SignedVolume : public Mul {
public:
    static constexpr std::array<const char*, 2> kInputNames{ { "sign", "volume" } };
};

// L1 queue imbalance, (bid_size - ask_size) / (bid_size + ask_size). The OFI
// kernel applied to resting queue sizes.
class QueueImbalance : public OFI {
public:
    static constexpr std::array<const char*, 2> kInputNames{ { "bid_size", "ask_size" } };
};

// Trailing-window sum of signed order flow. The RollingSum kernel.
class RollingOrderImbalance : public RollingSum {
public:
    using RollingSum::RollingSum;
    static constexpr std::array<const char*, 1> kInputNames{ { "signed_flow" } };
};

// Kyle's lambda over a trailing window: the slope of return on signed flow.
// The RollingBeta kernel with the inputs swapped.
class RollingKyleLambda : public detail::SwapTwoInputs<RollingBeta> {
public:
    using SwapTwoInputs::SwapTwoInputs;
    static constexpr std::array<const char*, 2> kInputNames{ { "signed_flow", "return_" } };
};

// Kyle's lambda, exponentially weighted. The EwBeta kernel with the inputs
// swapped.
class EwKyleLambda : public detail::SwapTwoInputs<EwBeta> {
public:
    using SwapTwoInputs::SwapTwoInputs;
    static constexpr std::array<const char*, 2> kInputNames{ { "signed_flow", "return_" } };
};

}  // namespace screamer

#endif  // SCREAMER_MICROSTRUCTURE_SYNONYMS_H
