#ifndef SCREAMER_CUM_MAX_H
#define SCREAMER_CUM_MAX_H

// CumMax: running maximum from sample 0 to the current sample. O(1) memory.
// Output is monotonically non-decreasing while samples are finite.
//
// NaN policy "ignore" (see docs/nan_policy.md): a NaN input is skipped --
// output is NaN at that index, the running maximum is unchanged. This
// differs from numpy.maximum.accumulate, which propagates NaN forever.
//
// Useful for high-water marks and as the building block of Drawdown
// (drawdown[t] = x[t] / CumMax(x)[t] - 1).

#include <cmath>
#include <limits>
#include "screamer/common/base.h"

namespace screamer {

class CumMax : public ScreamerBase {
public:
    CumMax() { reset(); }

    void reset() override {
        running_max_ = -std::numeric_limits<double>::infinity();
    }

    double process_scalar(double x) override {
        if (std::isnan(x)) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        if (x > running_max_) {
            running_max_ = x;
        }
        return running_max_;
    }

private:
    double running_max_ = -std::numeric_limits<double>::infinity();
};

}  // namespace screamer

#endif  // SCREAMER_CUM_MAX_H
