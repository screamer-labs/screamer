#ifndef SCREAMER_ROLLING_POLY2_H
#define SCREAMER_ROLLING_POLY2_H

// RollingPoly2: least-squares parabola y = a t^2 + b t + c through the window
// against the time index t = 0 (oldest) .. n - 1 (newest); returns the
// parabola at the newest sample (derivative_order 0), its slope there
// (derivative_order 1, 2 a (n - 1) + b) or its curvature (derivative_order 2,
// 2 a).
//
// The window keeps the sums of d = x - origin, t d and t^2 d about an origin
// near the window mean (detail::ShiftedWindowSums<2, 2>), and the fit uses the
// orthogonal form of detail::time_regression.h, so the result stays accurate
// when the level is large next to the spread. O(1) per step.

#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include "screamer/detail/start_policy.h"
#include "screamer/detail/time_regression.h"
#include "screamer/detail/window_moments_op.h"

namespace screamer {

class RollingPoly2 : public detail::WindowMomentsOp<RollingPoly2, 2, 2> {
public:
    RollingPoly2(int window_size, int derivative_order = 0, const std::string& start_policy = "strict")
        : WindowMomentsOp(window_size, detail::parse_start_policy(start_policy),
                          detail::parse_start_policy(start_policy)),
          derivative_order_(derivative_order),
          full_(detail::TimeParabola::of(static_cast<double>(window_size)))
    {
        if (window_size < 3) {
            throw std::invalid_argument("Window size must be 3 or more.");
        }
        if (derivative_order < 0 || derivative_order > 2) {
            throw std::invalid_argument("Derivative order must be 0 (endpoint), 1 (slope), or 2 (curvature).");
        }
        reset();
    }

    inline double value(const State& st, double) const {
        const std::size_t w = static_cast<std::size_t>(window_size_);
        if (st.count != w) {
            if (start_policy_ == detail::StartPolicy::Strict || st.count < 3) {
                return std::numeric_limits<double>::quiet_NaN();
            }
        }
        const detail::TimeParabola c = (st.count == w)
            ? full_ : detail::TimeParabola::of(static_cast<double>(st.count));
        const detail::ParabolaFit f = detail::parabola_fit(st.s[0], st.ts[0], st.ts[1], c);
        if (derivative_order_ == 2) return 2.0 * f.a;
        if (derivative_order_ == 1) return detail::parabola_end_slope(f, c);
        return st.origin + detail::parabola_end_d(st.s[0], f, c);
    }

private:
    const int derivative_order_;
    const detail::TimeParabola full_;
};

}  // namespace screamer

#endif  // SCREAMER_ROLLING_POLY2_H
