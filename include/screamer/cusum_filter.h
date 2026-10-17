#ifndef SCREAMER_CUSUM_FILTER_H
#define SCREAMER_CUSUM_FILTER_H

// CusumFilter: the symmetric CUSUM event filter (Lopez de Prado 2018,
// section 2.5.2.1). Two running sums of the input,
//     S+ = max(0, S+ + x - drift),   S- = min(0, S- + x + drift),
// give -1 when S- < -threshold and +1 when S+ > threshold; the sum that fired
// restarts from 0 (as in the book's code), otherwise the output is 0. At the
// step an event fires, the other sum is already 0, so resetting it would
// change nothing. The input
// is normally returns or price differences, and an event marks a cumulative
// move of more than `threshold` since the last event. 1 -> 1, O(1) per step.
// nan_policy: ignore (a NaN gives NaN and leaves both sums unchanged).

#include <limits>
#include <stdexcept>
#include "screamer/common/base.h"
#include "screamer/common/finite_settings.h"
#include "screamer/common/float_info.h"

namespace screamer {

class CusumFilter : public ScreamerBase {
public:
    explicit CusumFilter(double threshold, double drift = 0.0)
        : threshold_(threshold), drift_(drift)
    {
        require_finite("threshold", threshold);
        require_finite("drift", drift);
        if (!(threshold > 0.0)) throw std::invalid_argument("threshold must be positive.");
        if (drift < 0.0) throw std::invalid_argument("drift must be 0 or more.");
        reset();
    }

    void reset() override {
        pos_ = 0.0;
        neg_ = 0.0;
    }

private:
    double process_scalar(double x) override {
        if (isnan2(x)) return std::numeric_limits<double>::quiet_NaN();
        const double up = pos_ + x - drift_;
        const double down = neg_ + x + drift_;
        pos_ = up > 0.0 ? up : 0.0;
        neg_ = down < 0.0 ? down : 0.0;
        if (neg_ < -threshold_) {
            neg_ = 0.0;
            return -1.0;
        }
        if (pos_ > threshold_) {
            pos_ = 0.0;
            return 1.0;
        }
        return 0.0;
    }

    const double threshold_;
    const double drift_;
    double pos_ = 0.0;
    double neg_ = 0.0;
};

}  // namespace screamer

#endif  // SCREAMER_CUSUM_FILTER_H
