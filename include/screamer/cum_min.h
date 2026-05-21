#ifndef SCREAMER_CUM_MIN_H
#define SCREAMER_CUM_MIN_H

// CumMin: running minimum from sample 0 to the current sample. O(1) memory.
// Output is monotonically non-increasing while samples are finite.
//
// NaN policy "ignore" (see docs/nan_policy.md): a NaN input is skipped --
// output is NaN at that index, the running minimum is unchanged. This
// differs from numpy.minimum.accumulate, which propagates NaN forever.

#include <cmath>
#include <limits>
#include "screamer/common/base.h"

namespace screamer {

class CumMin : public ScreamerBase {
public:
    CumMin() { reset(); }

    void reset() override {
        running_min_ = std::numeric_limits<double>::infinity();
    }

    double process_scalar(double x) override {
        if (std::isnan(x)) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        if (x < running_min_) {
            running_min_ = x;
        }
        return running_min_;
    }

private:
    double running_min_ = std::numeric_limits<double>::infinity();
};

}  // namespace screamer

#endif  // SCREAMER_CUM_MIN_H
