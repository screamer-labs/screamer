#ifndef SCREAMER_STOCH_H
#define SCREAMER_STOCH_H

// Stoch: Stochastic oscillator (George Lane, 1950s). Returns the
// pair (%K, %D) per step:
//
//     raw_K[t]  = 100 * (close - L_n) / (H_n - L_n)
//     %K[t]     = SMA(raw_K, smooth_k)[t]
//     %D[t]     = SMA(%K,    d)[t]
//
// where H_n / L_n are the rolling max of high / min of low over
// `fastk_period`. With smooth_k = 1 this collapses to the "fast"
// Stochastic (Lane's original); with smooth_k = 3 it matches the
// "slow" Stochastic that talib.STOCH and most charts report.
//
// Composition: two detail::MonotonicDeque (for H_n / L_n) plus two
// detail::RollingMean instances (for the smooth_k and d smoothers).
// Amortised O(1) per step.
//
// Warmup: NaN-NaN until both %K and %D are valid, which happens at
// sample index (fastk_period + smooth_k + d - 3). Matches TA-Lib.
//
// Range-zero handling: when H_n == L_n over the period, raw_K is
// mathematically undefined. We return 0 there (matching TA-Lib's
// WILLR convention; STOCH itself uses 0 too).

#include <limits>
#include <stdexcept>
#include "screamer/common/functor_base.h"
#include "screamer/detail/monotonic_deque.h"
#include "screamer/detail/rolling_mean.h"

namespace screamer {

class Stoch : public FunctorBase<Stoch, 3, 2> {
public:
    Stoch(int fastk_period = 14, int smooth_k = 3, int d = 3)
        : fastk_period_(fastk_period),
          smooth_k_(smooth_k),
          d_(d),
          max_deque_(fastk_period),
          min_deque_(fastk_period),
          // Inner SMAs run "expanding" so they always return finite
          // values; this class controls warmup gating itself.
          smooth_k_mean_(static_cast<size_t>(smooth_k), "expanding"),
          d_mean_(static_cast<size_t>(d), "expanding")
    {
        if (fastk_period < 1 || smooth_k < 1 || d < 1) {
            throw std::invalid_argument(
                "fastk_period, smooth_k, and d must be >= 1.");
        }
    }

    void reset() override {
        max_deque_.reset();
        min_deque_.reset();
        smooth_k_mean_.reset();
        d_mean_.reset();
        n_seen_ = 0;
    }

    ResultTuple call(const InputArray& inputs) override {
        const double high  = inputs[0];
        const double low   = inputs[1];
        const double close = inputs[2];

        const double h_n = max_deque_.append(high);
        const double l_n = min_deque_.append(low);

        n_seen_++;

        // Stage 1: raw_K needs fastk_period samples.
        if (n_seen_ < fastk_period_) {
            return std::make_tuple(
                std::numeric_limits<double>::quiet_NaN(),
                std::numeric_limits<double>::quiet_NaN());
        }

        const double range = h_n - l_n;
        const double raw_k = (range <= 0.0)
            ? 0.0
            : 100.0 * (close - l_n) / range;

        // Stage 2: %K needs smooth_k samples of raw_K.
        const double slow_k = smooth_k_mean_.append(raw_k);
        if (n_seen_ < fastk_period_ + smooth_k_ - 1) {
            return std::make_tuple(
                std::numeric_limits<double>::quiet_NaN(),
                std::numeric_limits<double>::quiet_NaN());
        }

        // Stage 3: %D needs d samples of %K.
        const double slow_d = d_mean_.append(slow_k);
        if (n_seen_ < fastk_period_ + smooth_k_ + d_ - 2) {
            return std::make_tuple(
                std::numeric_limits<double>::quiet_NaN(),
                std::numeric_limits<double>::quiet_NaN());
        }

        return std::make_tuple(slow_k, slow_d);
    }

private:
    const int fastk_period_;
    const int smooth_k_;
    const int d_;
    detail::MaxDeque max_deque_;
    detail::MinDeque min_deque_;
    detail::RollingMean smooth_k_mean_;
    detail::RollingMean d_mean_;
    int n_seen_ = 0;
};

}  // namespace screamer

#endif
