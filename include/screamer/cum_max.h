#ifndef SCREAMER_CUM_MAX_H
#define SCREAMER_CUM_MAX_H

// CumMax: running maximum from sample 0 to the current sample. O(1) memory.
// Output is monotonically non-decreasing while samples are finite.
// NaN propagates: once an input is NaN, all subsequent outputs are NaN.
// Matches numpy.maximum.accumulate / numpy.cummax behavior.
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
        if (std::isnan(x) || std::isnan(running_max_)) {
            running_max_ = std::numeric_limits<double>::quiet_NaN();
            return running_max_;
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
