#ifndef SCREAMER_EXPANDING_SLOPE_H
#define SCREAMER_EXPANDING_SLOPE_H

// ExpandingSlope: OLS slope of y against an implicit time axis x = 0, 1, ..., n-1
// over the whole history since the last reset(). O(1) memory, no window.
//
//   slope = sum (t - tbar)(y - mean y) / sum (t - tbar)^2,  tbar = (n - 1) / 2
//
// The sums of d = y - origin and t d are kept about an origin that follows the
// mean (detail::ShiftedPowerSums<2, true>), and the fit is taken about the mean
// time (detail::time_regression.h), so the slope stays accurate when the level
// is large next to the spread. Undefined (NaN) for n < 2 (a line through a
// single point has no slope).
//
// NaN policy "ignore": a NaN input is skipped -- output is NaN at that index
// and internal state (including the implicit time index) is unchanged.

#include <limits>
#include "screamer/detail/running_moments_op.h"
#include "screamer/detail/time_regression.h"

namespace screamer {

class ExpandingSlope : public detail::RunningMomentsOp<ExpandingSlope, 2, false, true> {
public:
    ExpandingSlope() { reset(); }

    inline double value(const State& st, double) const {
        const double n = st.sums.w;
        if (n < 2.0) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        const double num = detail::centered_time_sum(st.sums.s1, st.sums.ts, 0.5 * (n - 1.0));
        const double nn = n * n;
        const double stt = n * (nn - 1.0) / 12.0;  // sum (t - tbar)^2
        return num / stt;
    }
};

}  // namespace screamer

#endif  // SCREAMER_EXPANDING_SLOPE_H
