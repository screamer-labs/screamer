#ifndef SCREAMER_CUM_SUM_H
#define SCREAMER_CUM_SUM_H

// CumSum: running sum from sample 0 to the current sample. O(1) memory.
//
// NaN policy "ignore" (see docs/nan_policy.md): a NaN input is skipped --
// output is NaN at that index, the running sum is unchanged. This matches
// pandas Series.cumsum(skipna=True) (the default) and differs from
// numpy.cumsum.

#include <cmath>
#include <limits>
#include "screamer/common/base.h"

namespace screamer {

class CumSum : public ScreamerBase {
public:
    CumSum() { reset(); }

    void reset() override { sum_ = 0.0; }

    double process_scalar(double x) override {
        if (std::isnan(x)) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        sum_ += x;
        return sum_;
    }

private:
    double sum_ = 0.0;
};

}  // namespace screamer

#endif  // SCREAMER_CUM_SUM_H
