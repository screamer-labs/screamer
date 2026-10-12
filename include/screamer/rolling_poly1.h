#ifndef SCREAMER_ROLLING_POLY1_H
#define SCREAMER_ROLLING_POLY1_H

// RollingPoly1: least-squares line through the window against the time index
// t = 0 (oldest) .. n - 1 (newest); returns the line at the newest sample
// (derivative_order 0) or its slope (derivative_order 1).
//
// The window keeps the sums of d = x - origin and t d about an origin near the
// window mean (detail::ShiftedWindowSums<2, 1>), and the fit is taken about the
// mean time (detail::time_regression.h), so the result stays accurate when the
// level is large next to the spread. O(1) per step.

#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include "screamer/detail/start_policy.h"
#include "screamer/detail/time_regression.h"
#include "screamer/detail/window_moments_op.h"

namespace screamer {

class RollingPoly1 : public detail::WindowMomentsOp<RollingPoly1, 2, 1> {
public:
    RollingPoly1(int window_size, int derivative_order = 0, const std::string& start_policy = "strict")
        : WindowMomentsOp(window_size, detail::parse_start_policy(start_policy),
                          detail::parse_start_policy(start_policy)),
          derivative_order_(derivative_order),
          full_(detail::TimeLine::of(static_cast<double>(window_size)))
    {
        if (window_size < 2) {
            throw std::invalid_argument("Window size must be 2 or more.");
        }
        if (derivative_order != 0 && derivative_order != 1) {
            throw std::invalid_argument("Derivative order must be 0 (endpoint) or 1 (slope).");
        }
        reset();
    }

    inline double value(const State& st, double) const {
        const std::size_t w = static_cast<std::size_t>(window_size_);
        if (st.count != w) {
            if (start_policy_ == detail::StartPolicy::Strict || st.count < 2) {
                return std::numeric_limits<double>::quiet_NaN();
            }
        }
        const detail::TimeLine c = (st.count == w)
            ? full_ : detail::TimeLine::of(static_cast<double>(st.count));
        const double slope = detail::line_slope(st.s[0], st.ts[0], c);
        if (derivative_order_ == 1) return slope;
        // The line at t = n - 1, that is tbar after the mean time.
        return st.origin + detail::line_value_d(st.s[0], slope, c.tbar, c);
    }

private:
    const int derivative_order_;
    const detail::TimeLine full_;
};

}  // namespace screamer

#endif  // SCREAMER_ROLLING_POLY1_H
