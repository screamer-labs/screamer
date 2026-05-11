#ifndef SCREAMER_ROLLING_HIT_RATE_H
#define SCREAMER_ROLLING_HIT_RATE_H

// RollingHitRate: fraction of strictly-positive samples in the
// trailing window.
//
//     HitRate[t] = (1 / w) * count(x_i > 0,  i in window)
//
// Output is in [0, 1]. Useful for "what fraction of recent bars
// went my way?" -- strategy-evaluation streams, signal-quality
// dashboards.
//
// 1 -> 1. Composes detail::RollingSum over an indicator (x > 0)
// stream. O(1) per step.

#include <limits>
#include <stdexcept>
#include "screamer/common/base.h"
#include "screamer/detail/rolling_sum.h"

namespace screamer {

class RollingHitRate : public ScreamerBase {
public:
    explicit RollingHitRate(int window_size)
        : window_size_(window_size),
          hits_(window_size, "expanding")
    {
        if (window_size < 1) {
            throw std::invalid_argument("Window size must be positive.");
        }
    }

    void reset() override {
        hits_.reset();
        n_seen_ = 0;
    }

    double process_scalar(double x) override {
        const double indicator = (x > 0.0) ? 1.0 : 0.0;
        const double total = hits_.append(indicator);
        if (n_seen_ < window_size_) {
            n_seen_++;
            if (n_seen_ < window_size_) {
                return std::numeric_limits<double>::quiet_NaN();
            }
        }
        return total / window_size_;
    }

private:
    const int window_size_;
    detail::RollingSum hits_;
    int n_seen_ = 0;
};

}  // namespace screamer

#endif
