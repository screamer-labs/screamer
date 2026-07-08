#ifndef SCREAMER_EXPANDING_VAR_H
#define SCREAMER_EXPANDING_VAR_H

// ExpandingVar: sample variance (ddof=1) over the whole history since the last
// reset(). O(1) memory, no window, no start_policy. Mirrors RollingVar's
// formula (via var_from_stats) but keeps unbounded running sums.
//
// Convention: ddof=1 (sample variance), matching RollingVar / RollingStd and
// pandas Series.expanding().var() (default ddof=1). Undefined (NaN) for n < 2.
//
// NaN policy "ignore" (see docs/nan_policy.md): a NaN input is skipped --
// output is NaN at that index and internal state is unchanged.

#include <cmath>
#include <limits>
#include "screamer/common/base.h"
#include "screamer/common/math.h"

namespace screamer {

class ExpandingVar : public ScreamerBase {
public:
    ExpandingVar() { reset(); }

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
        return var;
    }

private:
    double sum_x_ = 0.0;
    double sum_xx_ = 0.0;
    long n_ = 0;
};

}  // namespace screamer

#endif  // SCREAMER_EXPANDING_VAR_H
