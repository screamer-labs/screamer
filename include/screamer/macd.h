#ifndef SCREAMER_MACD_H
#define SCREAMER_MACD_H

// MACD: Moving Average Convergence Divergence (Gerald Appel, late 1970s).
//
//   ema_fast[t]   = EwMean(span=fast)(x)[t]
//   ema_slow[t]   = EwMean(span=slow)(x)[t]
//   macd[t]       = ema_fast[t] - ema_slow[t]
//   signal[t]     = EwMean(span=signal)(macd)[t]
//   histogram[t]  = macd[t] - signal[t]
//
// Returns the triple (macd, signal, histogram) per step. Pure
// composition of three EwMean instances; O(1) per step. The
// underlying EMA is pandas's adjust=True form (the
// statistically-clean weighted mean we use everywhere -- see
// docs/conventions.md).
//
// Defaults match TA-Lib's: fast=12, slow=26, signal=9.

#include <stdexcept>
#include "screamer/common/functor_base.h"
#include "screamer/ew_mean.h"

namespace screamer {

class MACD : public FunctorBase<MACD, 1, 3> {
public:
    MACD(int fast = 12, int slow = 26, int signal = 9)
        : ema_fast_(std::nullopt, static_cast<double>(fast),
                    std::nullopt, std::nullopt),
          ema_slow_(std::nullopt, static_cast<double>(slow),
                    std::nullopt, std::nullopt),
          ema_signal_(std::nullopt, static_cast<double>(signal),
                      std::nullopt, std::nullopt)
    {
        if (fast < 1 || slow < 1 || signal < 1) {
            throw std::invalid_argument("fast, slow, and signal must be positive.");
        }
        if (fast >= slow) {
            throw std::invalid_argument("fast must be strictly less than slow.");
        }
    }

    void reset() override {
        ema_fast_.reset();
        ema_slow_.reset();
        ema_signal_.reset();
    }

    ResultTuple call(const InputArray& inputs) override {
        const double x = inputs[0];
        const double f = ema_fast_.process_scalar(x);
        const double s = ema_slow_.process_scalar(x);
        const double macd = f - s;
        const double signal = ema_signal_.process_scalar(macd);
        const double histogram = macd - signal;
        return std::make_tuple(macd, signal, histogram);
    }

private:
    EwMean ema_fast_;
    EwMean ema_slow_;
    EwMean ema_signal_;
};

}  // namespace screamer

#endif
