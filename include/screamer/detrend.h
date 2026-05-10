#ifndef SCREAMER_DETREND_H
#define SCREAMER_DETREND_H

// Detrend: subtract a rolling mean of the input.
//
//     y[t] = x[t] - RollingMean(window)(x)[t]
//
// One detail::RollingMean buffer under the hood, O(1) per step.
// During warmup the rolling mean is NaN under start_policy="strict",
// which propagates to the output.

#include <string>
#include "screamer/common/base.h"
#include "screamer/detail/rolling_mean.h"

namespace screamer {

class Detrend : public ScreamerBase {
public:
    Detrend(int window_size, const std::string& start_policy = "strict")
        : rolling_mean_(window_size, start_policy)
    {}

    void reset() override {
        rolling_mean_.reset();
    }

    double process_scalar(double x) override {
        return x - rolling_mean_.append(x);
    }

private:
    detail::RollingMean rolling_mean_;
};

}  // namespace screamer

#endif  // SCREAMER_DETREND_H
