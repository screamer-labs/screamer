#ifndef SCREAMER_EXPANDING_MEAN_H
#define SCREAMER_EXPANDING_MEAN_H

// ExpandingMean: running mean over the whole history since the last reset().
// O(1) memory, no window, no start_policy. Mirrors RollingMean but keeps an
// unbounded running sum instead of a windowed detail::RollingSum.
//
// NaN policy "ignore" (see docs/nan_policy.md): a NaN input is skipped --
// output is NaN at that index and the running sum / count are unchanged.
// Matches pandas Series.expanding().mean().

#include <cmath>
#include <limits>
#include "screamer/common/base.h"

namespace screamer {

class ExpandingMean : public ScreamerBase {
public:
    ExpandingMean() { reset(); }

    void reset() override {
        sum_x_ = 0.0;
        n_ = 0;
    }

    double process_scalar(double x) override {
        if (std::isnan(x)) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        sum_x_ += x;
        ++n_;
        return sum_x_ / n_;
    }

private:
    double sum_x_ = 0.0;
    long n_ = 0;
};

}  // namespace screamer

#endif  // SCREAMER_EXPANDING_MEAN_H
