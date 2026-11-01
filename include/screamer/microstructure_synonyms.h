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
// them as keywords, and saves under its own name (serial::Renamed).

#include <array>
#include <cstddef>
#include <memory>
#include "screamer/arithmetic.h"
#include "screamer/ofi.h"
#include "screamer/rolling_sum.h"
#include "screamer/rolling_beta.h"
#include "screamer/ew_beta.h"

namespace screamer {

namespace detail {

// A two-input operator that passes its inputs to Base in the reverse order.
// Every path into the kernel is one of the overrides: call() for one event
// (and EvalOp::eval, which calls it), and eval_column() and
// eval_column_resume() for a batch column. The column swaps its two input pointers and strides, so the batch
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

    void eval_column_resume(const double* const* in, const std::ptrdiff_t* in_stride,
                            std::size_t T, double* out, std::ptrdiff_t out_stride) override {
        const double* const swapped[2] = { in[1], in[0] };
        const std::ptrdiff_t strides[2] = { in_stride[1], in_stride[0] };
        Base::eval_column_resume(swapped, strides, T, out, out_stride);
    }

    // A copy that keeps the swap; Base's would drop it.
    std::unique_ptr<EvalOp> clone_fresh() const override {
        auto p = std::make_unique<SwapTwoInputs>(*this);
        p->reset();
        return p;
    }
};

}  // namespace detail

// Signed order flow, sign * volume. The Mul kernel.
class SignedVolume : public serial::Renamed<SignedVolume, Mul> {
public:
    static constexpr const char* kTypeName = "SignedVolume";
    static constexpr std::array<const char*, 2> kInputNames{ { "sign", "volume" } };
};

// L1 queue imbalance, (bid_size - ask_size) / (bid_size + ask_size). The OFI
// kernel applied to resting queue sizes.
class QueueImbalance : public serial::Renamed<QueueImbalance, OFI> {
public:
    static constexpr const char* kTypeName = "QueueImbalance";
    static constexpr std::array<const char*, 2> kInputNames{ { "bid_size", "ask_size" } };
};

// Trailing-window sum of signed order flow. The RollingSum kernel.
class RollingOrderImbalance : public serial::Renamed<RollingOrderImbalance, RollingSum> {
public:
    static constexpr const char* kTypeName = "RollingOrderImbalance";
    using Renamed::Renamed;
    static constexpr std::array<const char*, 1> kInputNames{ { "signed_flow" } };
};

// Kyle's lambda over a trailing window: the slope of return on signed flow.
// The RollingBeta kernel with the inputs swapped.
class RollingKyleLambda
    : public serial::Renamed<RollingKyleLambda, detail::SwapTwoInputs<RollingBeta>> {
public:
    static constexpr const char* kTypeName = "RollingKyleLambda";
    using Renamed::Renamed;
    static constexpr std::array<const char*, 2> kInputNames{ { "signed_flow", "return_" } };
};

// Kyle's lambda, exponentially weighted. The EwBeta kernel with the inputs
// swapped.
class EwKyleLambda : public serial::Renamed<EwKyleLambda, detail::SwapTwoInputs<EwBeta>> {
public:
    static constexpr const char* kTypeName = "EwKyleLambda";
    using Renamed::Renamed;
    static constexpr std::array<const char*, 2> kInputNames{ { "signed_flow", "return_" } };
};

}  // namespace screamer

#endif  // SCREAMER_MICROSTRUCTURE_SYNONYMS_H
