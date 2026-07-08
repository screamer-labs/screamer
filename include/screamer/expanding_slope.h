#ifndef SCREAMER_EXPANDING_SLOPE_H
#define SCREAMER_EXPANDING_SLOPE_H

// ExpandingSlope: OLS slope of y against an implicit time axis x = 0, 1, ..., n-1
// over the whole history since the last reset(). O(1) memory, no window.
// Mirrors RollingPoly1's closed-form slope (derivative_order = 1) but drops the
// DelayBuffer and windowing in favor of unbounded running sums.
//
// With x = 0..n-1:
//   Sx  = n(n-1)/2
//   Sxx = (n-1)n(2n-1)/6
//   a   = (n*Sxy - Sx*Sy) / (n*Sxx - Sx*Sx)
// Undefined (NaN) for n < 2 (a line through a single point has no slope).
//
// NaN policy "ignore": a NaN input is skipped -- output is NaN at that index
// and internal state (including the implicit time index) is unchanged.

#include <cmath>
#include <limits>
#include "screamer/common/base.h"

namespace screamer {

class ExpandingSlope : public ScreamerBase {
public:
    ExpandingSlope() { reset(); }

    void reset() override {
        sum_y_ = 0.0;
        sum_xy_ = 0.0;
        n_ = 0;
    }

    double process_scalar(double y) override {
        if (std::isnan(y)) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        // The new sample sits at implicit time index n_ (0-based) before the
        // count is bumped, so only finite samples advance the time axis.
        sum_xy_ += static_cast<double>(n_) * y;
        sum_y_ += y;
        ++n_;

        if (n_ < 2) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        double n = static_cast<double>(n_);
        double sum_x = (n - 1.0) * n / 2.0;
        double sum_xx = (n - 1.0) * n * (2.0 * n - 1.0) / 6.0;
        return (n * sum_xy_ - sum_x * sum_y_) / (n * sum_xx - sum_x * sum_x);
    }

private:
    double sum_y_ = 0.0;
    double sum_xy_ = 0.0;
    long n_ = 0;
};

}  // namespace screamer

#endif  // SCREAMER_EXPANDING_SLOPE_H
