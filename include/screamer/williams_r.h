#ifndef SCREAMER_WILLIAMS_R_H
#define SCREAMER_WILLIAMS_R_H

// WilliamsR: Williams %R (Larry Williams 1973). Normalised position of
// the close within the recent (high, low) range, scaled to [-100, 0]:
// 0 means the close is at the period high, -100 means it is at the
// period low.
//
//     %R[t] = -100 * (high_n - close) / (high_n - low_n)
//
// where high_n / low_n are the rolling max / min of high and low over
// the period. A 3 -> 1 functor (FunctorBase<_, 3, 1>): inputs are
// (high, low, close) in TA-Lib's argument order.
//
// Composition: two detail::MonotonicDeque instances (the same
// primitive RollingMin / RollingMax / RollingArgmin / RollingArgmax /
// RollingRange use). Amortised O(1) per step.
//
// Warmup: NaN for the first window_size - 1 samples; first valid
// output at sample index window_size - 1 (TA-Lib's convention).
//
// Range-zero handling: when high_n == low_n (a perfectly flat
// segment), the formula is mathematically undefined. TA-Lib returns 0
// in that case; we follow.

#include <limits>
#include <stdexcept>
#include "screamer/common/functor_base.h"
#include "screamer/detail/monotonic_deque.h"

namespace screamer {

class WilliamsR : public FunctorBase<WilliamsR, 3, 1> {
public:
    explicit WilliamsR(int window_size = 14)
        : window_size_(window_size),
          max_deque_(window_size),
          min_deque_(window_size)
    {
        if (window_size < 1) {
            throw std::invalid_argument("Window size must be at least 1.");
        }
    }

    void reset() override {
        max_deque_.reset();
        min_deque_.reset();
        n_seen_ = 0;
    }

    ResultTuple call(const InputArray& inputs) override {
        const double high  = inputs[0];
        const double low   = inputs[1];
        const double close = inputs[2];

        const double high_n = max_deque_.append(high);
        const double low_n  = min_deque_.append(low);

        if (n_seen_ < window_size_) {
            n_seen_++;
        }
        if (n_seen_ < window_size_) {
            return std::numeric_limits<double>::quiet_NaN();
        }

        const double range = high_n - low_n;
        if (range <= 0.0) {
            return 0.0;
        }
        return -100.0 * (high_n - close) / range;
    }

private:
    const int window_size_;
    detail::MaxDeque max_deque_;
    detail::MinDeque min_deque_;
    int n_seen_ = 0;
};

}  // namespace screamer

#endif
