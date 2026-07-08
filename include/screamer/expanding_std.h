#ifndef SCREAMER_EXPANDING_STD_H
#define SCREAMER_EXPANDING_STD_H

// ExpandingStd: sample standard deviation (ddof=1) over the whole history
// since the last reset(). sqrt of ExpandingVar. O(1) memory, no window.
//
// Convention: ddof=1, matching RollingStd and pandas
// Series.expanding().std() (default ddof=1). Undefined (NaN) for n < 2.
//
// NaN policy "ignore": a NaN input is skipped -- output is NaN at that index
// and internal state is unchanged.

#include <cmath>
#include <limits>
#include "screamer/common/base.h"
#include "screamer/common/math.h"

namespace screamer {

class ExpandingStd : public ScreamerBase {
public:
    ExpandingStd() { reset(); }

    void reset() override {
        sum_x_ = 0.0;
        sum_xx_ = 0.0;
        n_ = 0;
    }

    double process_scalar(double x) override {
        if (std::isnan(x)) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        sum_x_ += x;
        sum_xx_ += x * x;
        ++n_;
        if (n_ < 2) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        double var;
        var_from_stats(sum_x_, sum_xx_, static_cast<int>(n_), var);
        return std::sqrt(var);
    }

private:
    double sum_x_ = 0.0;
    double sum_xx_ = 0.0;
    long n_ = 0;
};

}  // namespace screamer

#endif  // SCREAMER_EXPANDING_STD_H
