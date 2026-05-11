#ifndef SCREAMER_STOCH_RSI_H
#define SCREAMER_STOCH_RSI_H

// StochRSI: Stochastic of RSI (Chande & Kroll, 1994). Applies the
// Stochastic formula to an RSI series instead of price:
//
//     RSI[t]   = RollingRSI(rsi_period, method="wilder")(x)[t]
//     raw_K    = 100 * (RSI - rolling_min(RSI, stoch_period))
//                    / (rolling_max(RSI, stoch_period) - rolling_min(...))
//     %K[t]    = SMA(raw_K, smooth_k)
//     %D[t]    = SMA(%K,    d)
//
// 1 -> 2 functor (one input, returns (%K, %D)).
//
// smooth_k = 1 (default, matching TA-Lib's STOCHRSI) collapses to
// the "fast" form where %K == raw_K; smooth_k >= 2 yields the "slow"
// form some platforms expose.
//
// Composition: RollingRSI (Wilder) + two detail::MonotonicDeque +
// two detail::RollingMean. Amortised O(1) per step.

#include <limits>
#include <stdexcept>
#include "screamer/common/functor_base.h"
#include "screamer/detail/monotonic_deque.h"
#include "screamer/detail/rolling_mean.h"
#include "screamer/rolling_rsi.h"

namespace screamer {

class StochRSI : public FunctorBase<StochRSI, 1, 2> {
public:
    StochRSI(int rsi_period = 14,
             int stoch_period = 14,
             int smooth_k = 1,
             int d = 3)
        : rsi_period_(rsi_period),
          stoch_period_(stoch_period),
          smooth_k_(smooth_k),
          d_(d),
          rsi_(rsi_period, "wilder", "strict"),
          max_deque_(stoch_period),
          min_deque_(stoch_period),
          smooth_k_mean_(static_cast<size_t>(smooth_k), "expanding"),
          d_mean_(static_cast<size_t>(d), "expanding")
    {
        if (rsi_period < 2 || stoch_period < 1 || smooth_k < 1 || d < 1) {
            throw std::invalid_argument(
                "rsi_period must be >= 2 and other periods must be >= 1.");
        }
    }

    void reset() override {
        rsi_.reset();
        max_deque_.reset();
        min_deque_.reset();
        smooth_k_mean_.reset();
        d_mean_.reset();
        n_valid_rsi_ = 0;
    }

    ResultTuple call(const InputArray& inputs) override {
        const double rsi_val = rsi_.process_scalar(inputs[0]);

        // RSI is NaN during its warmup (rsi_period samples). Don't feed
        // the downstream stages until RSI is valid.
        if (std::isnan(rsi_val)) {
            return std::make_tuple(
                std::numeric_limits<double>::quiet_NaN(),
                std::numeric_limits<double>::quiet_NaN());
        }

        const double rsi_max = max_deque_.append(rsi_val);
        const double rsi_min = min_deque_.append(rsi_val);

        n_valid_rsi_++;

        // Need stoch_period RSI values to fill the deques.
        if (n_valid_rsi_ < stoch_period_) {
            return std::make_tuple(
                std::numeric_limits<double>::quiet_NaN(),
                std::numeric_limits<double>::quiet_NaN());
        }

        const double range = rsi_max - rsi_min;
        const double raw_k = (range <= 0.0)
            ? 0.0
            : 100.0 * (rsi_val - rsi_min) / range;

        const double slow_k = smooth_k_mean_.append(raw_k);
        if (n_valid_rsi_ < stoch_period_ + smooth_k_ - 1) {
            return std::make_tuple(
                std::numeric_limits<double>::quiet_NaN(),
                std::numeric_limits<double>::quiet_NaN());
        }

        const double slow_d = d_mean_.append(slow_k);
        if (n_valid_rsi_ < stoch_period_ + smooth_k_ + d_ - 2) {
            return std::make_tuple(
                std::numeric_limits<double>::quiet_NaN(),
                std::numeric_limits<double>::quiet_NaN());
        }
        return std::make_tuple(slow_k, slow_d);
    }

private:
    const int rsi_period_;
    const int stoch_period_;
    const int smooth_k_;
    const int d_;
    RollingRSI rsi_;
    detail::MaxDeque max_deque_;
    detail::MinDeque min_deque_;
    detail::RollingMean smooth_k_mean_;
    detail::RollingMean d_mean_;
    int n_valid_rsi_ = 0;
};

}  // namespace screamer

#endif
