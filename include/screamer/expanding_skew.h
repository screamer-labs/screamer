#ifndef SCREAMER_EXPANDING_SKEW_H
#define SCREAMER_EXPANDING_SKEW_H

// ExpandingSkew: bias-corrected sample skewness (G1) over the whole history
// since the last reset(). O(1) memory, no window. Mirrors RollingSkew (via
// skew_from_stats / skew_n_const) but keeps unbounded running sums.
//
// Convention: the adjusted Fisher-Pearson standardized moment G1, identical to
// RollingSkew and pandas Series.expanding().skew(). Undefined (NaN) for n < 3.
//
// NaN policy "ignore": a NaN input is skipped -- output is NaN at that index
// and internal state is unchanged.

#include <cmath>
#include <limits>
#include "screamer/common/base.h"
#include "screamer/common/math.h"

namespace screamer {

class ExpandingSkew : public ScreamerBase {
public:
    ExpandingSkew() { reset(); }

    void reset() override {
        sum_x_ = 0.0;
        sum_xx_ = 0.0;
        sum_xxx_ = 0.0;
        n_ = 0;
    }

    double process_scalar(double x) override {
        if (std::isnan(x)) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        double x2 = x * x;
        sum_x_ += x;
        sum_xx_ += x2;
        sum_xxx_ += x2 * x;
        ++n_;
        if (n_ < 3) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        double c0;
        skew_n_const(static_cast<int>(n_), c0);
        double skew;
        skew_from_stats(sum_x_, sum_xx_, sum_xxx_, c0, static_cast<int>(n_), skew);
        return skew;
    }

private:
    double sum_x_ = 0.0;
    double sum_xx_ = 0.0;
    double sum_xxx_ = 0.0;
    long n_ = 0;
};

}  // namespace screamer

#endif  // SCREAMER_EXPANDING_SKEW_H
