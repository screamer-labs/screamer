#ifndef SCREAMER_ROLLING_MEDIAN_AD_H
#define SCREAMER_ROLLING_MEDIAN_AD_H

// RollingMedianAD: rolling *median* absolute deviation,
//
//     MedianAD[t] = median_{i in window} | x_i - median(window) |
//
// This is the robust scale estimate (unlike RollingMad, which is the *mean*
// absolute deviation and is not robust to outliers). It is the scale primitive
// used by the Hampel and ImpulseClip despikers. Returns the raw MAD; callers
// that want a Gaussian-consistent standard-deviation estimate scale by 1.4826.
//
// O(W) per step (two nth_element passes over the window), matching how
// RollingMad accepts O(W): the median shifts every step, so the absolute
// deviations cannot be maintained incrementally.

#include <limits>
#include <stdexcept>
#include <string>

#include "screamer/common/base.h"
#include "screamer/common/float_info.h"
#include "screamer/detail/robust_window.h"
#include "screamer/detail/start_policy.h"

namespace screamer {

class RollingMedianAD : public ScreamerBase {
public:
    RollingMedianAD(int window_size, const std::string& start_policy = "strict")
        : start_policy_(detail::parse_start_policy(start_policy)),
          window_(window_size)
    {
        if (window_size <= 0) {
            throw std::invalid_argument("Window size must be positive.");
        }
        reset();
    }

    void reset() override {
        window_.reset();
        if (start_policy_ == detail::StartPolicy::Zero) {
            window_.prefill_zero();
        }
    }

private:
    double process_scalar(double newValue) override {
        // NaN policy "ignore": leave the window untouched and emit NaN.
        if (isnan2(newValue)) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        window_.push(newValue);
        if (start_policy_ == detail::StartPolicy::Strict && !window_.full()) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        return window_.mad();
    }

    const detail::StartPolicy start_policy_;
    detail::RobustWindow window_;
};

}  // namespace screamer

#endif  // SCREAMER_ROLLING_MEDIAN_AD_H
