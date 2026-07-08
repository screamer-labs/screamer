#ifndef SCREAMER_EXPANDING_KURT_H
#define SCREAMER_EXPANDING_KURT_H

// ExpandingKurt: bias-corrected sample excess kurtosis (Fisher) over the whole
// history since the last reset(). O(1) memory, no window. Mirrors RollingKurt
// (via kurt_from_stats / kurt_n_const) but keeps unbounded running sums.
//
// Convention: the bias-corrected Fisher excess kurtosis, identical to
// RollingKurt and pandas Series.expanding().kurt(). Undefined (NaN) for n < 4.
//
// NaN policy "ignore": a NaN input is skipped -- output is NaN at that index
// and internal state is unchanged.

#include <cmath>
#include <limits>
#include "screamer/common/base.h"
#include "screamer/common/math.h"

namespace screamer {

class ExpandingKurt : public ScreamerBase {
public:
    ExpandingKurt() { reset(); }

    void reset() override {
        sum_x_ = 0.0;
        sum_xx_ = 0.0;
        sum_xxx_ = 0.0;
        sum_xxxx_ = 0.0;
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
        sum_xxxx_ += x2 * x2;
        ++n_;
        if (n_ < 4) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        double c0, c1, c2;
        kurt_n_const(static_cast<int>(n_), c0, c1, c2);
        double kurt;
        kurt_from_stats(sum_x_, sum_xx_, sum_xxx_, sum_xxxx_, c0, c1, c2,
                        static_cast<int>(n_), kurt);
        return kurt;
    }

private:
    double sum_x_ = 0.0;
    double sum_xx_ = 0.0;
    double sum_xxx_ = 0.0;
    double sum_xxxx_ = 0.0;
    long n_ = 0;
};

}  // namespace screamer

#endif  // SCREAMER_EXPANDING_KURT_H
