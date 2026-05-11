#ifndef SCREAMER_DONCHIAN_CHANNELS_H
#define SCREAMER_DONCHIAN_CHANNELS_H

// DonchianChannels (Richard Donchian). Trend-following envelope:
//
//     lower[t] = min(low,  window)
//     upper[t] = max(high, window)
//     mid[t]   = (lower + upper) / 2
//
// 2 -> 3 functor over (high, low). Returns (lower, mid, upper) per
// step. First valid output at sample index `window_size - 1`.
//
// Composition: two detail::MonotonicDeque -- same primitive used by
// RollingMin / Max / MinMax / Argmin / Argmax / Range / WilliamsR /
// Stoch. Amortised O(1) per step.

#include <limits>
#include <stdexcept>
#include <tuple>
#include "screamer/common/functor_base.h"
#include "screamer/detail/monotonic_deque.h"

namespace screamer {

class DonchianChannels : public FunctorBase<DonchianChannels, 2, 3> {
public:
    explicit DonchianChannels(int window_size = 20)
        : window_size_(window_size),
          max_deque_(window_size),
          min_deque_(window_size)
    {
        if (window_size < 1) {
            throw std::invalid_argument("Window size must be positive.");
        }
    }

    void reset() override {
        max_deque_.reset();
        min_deque_.reset();
        n_seen_ = 0;
    }

    ResultTuple call(const InputArray& inputs) override {
        const double high = inputs[0];
        const double low  = inputs[1];
        const double upper = max_deque_.append(high);
        const double lower = min_deque_.append(low);
        if (n_seen_ < window_size_) {
            n_seen_++;
        }
        if (n_seen_ < window_size_) {
            const double nan = std::numeric_limits<double>::quiet_NaN();
            return std::make_tuple(nan, nan, nan);
        }
        return std::make_tuple(lower, 0.5 * (lower + upper), upper);
    }

private:
    const int window_size_;
    detail::MaxDeque max_deque_;
    detail::MinDeque min_deque_;
    int n_seen_ = 0;
};

}  // namespace screamer

#endif
