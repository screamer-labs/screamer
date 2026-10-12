#ifndef SCREAMER_ROLLING_TSF_H
#define SCREAMER_ROLLING_TSF_H

// RollingTSF: Time-Series Forecast (TA-Lib's TSF). Linear regression of y
// against the time index over the trailing window, projected one step ahead.
//
//     For the window {(0, y_{t-n+1}), (1, y_{t-n+2}), ..., (n-1, y_t)}:
//     fit y = slope * t + intercept,
//     TSF[t] = slope * n + intercept   (the line at the next bar)
//
// Equivalently TSF[t] = LINEARREG[t] + LINEARREG_SLOPE[t] in TA-Lib's terms.
//
// The window keeps the sums of d = y - origin and t d about an origin near the
// window mean (detail::ShiftedWindowSums<2, 1>), and the fit is taken about the
// mean time (detail::time_regression.h), so the result stays accurate when the
// level is large next to the spread. O(1) per step. First valid at sample
// index window_size - 1.

#include <cstddef>
#include <limits>
#include <stdexcept>
#include "screamer/detail/start_policy.h"
#include "screamer/detail/time_regression.h"
#include "screamer/detail/window_moments_op.h"

namespace screamer {

class RollingTSF : public detail::WindowMomentsOp<RollingTSF, 2, 1> {
public:
    explicit RollingTSF(int window_size)
        : WindowMomentsOp(window_size, detail::StartPolicy::Strict, detail::StartPolicy::Strict),
          full_(detail::TimeLine::of(static_cast<double>(window_size))),
          ahead_(0.5 * (static_cast<double>(window_size) + 1.0))
    {
        if (window_size < 2) {
            throw std::invalid_argument("Window size must be at least 2.");
        }
        reset();
    }

    inline double value(const State& st, double) const {
        if (st.count != static_cast<std::size_t>(window_size_)) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        const double slope = detail::line_slope(st.s[0], st.ts[0], full_);
        // The line at t = n, that is (n + 1) / 2 after the mean time.
        return st.origin + detail::line_value_d(st.s[0], slope, ahead_, full_);
    }

private:
    const detail::TimeLine full_;
    const double ahead_;  // n - tbar
};

}  // namespace screamer

#endif  // SCREAMER_ROLLING_TSF_H
