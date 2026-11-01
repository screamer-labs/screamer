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

#include <cmath>
#include <limits>
#include <stdexcept>
#include <tuple>
#include "screamer/common/functor_base.h"
#include "screamer/detail/monotonic_deque.h"

namespace screamer {

class DonchianChannels : public FunctorBase<DonchianChannels, 2, 3> {
public:
    static constexpr const char* kTypeName = "DonchianChannels";
    static constexpr int kStateVersion = 1;

    template <class Ar>
    void settings(Ar& ar) {
        ar("window_size", window_size_);
    }

    template <class Ar>
    void state(Ar& ar) {
        ar.object("max", max_deque_);
        ar.object("min", min_deque_);
        ar("n_seen", n_seen_);
        ar.check(max_deque_.samples_seen() == min_deque_.samples_seen(),
                 "max and min must have seen the same samples");
        ar.check(n_seen_ >= 0 && n_seen_ <= window_size_, "n_seen must lie in 0 .. window_size");
    }

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
        if (std::isnan(high) || std::isnan(low)) {
            const double nan = std::numeric_limits<double>::quiet_NaN();
            return std::make_tuple(nan, nan, nan);
        }
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
