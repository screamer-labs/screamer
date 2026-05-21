#ifndef SCREAMER_ROLLING_RANGE_H
#define SCREAMER_ROLLING_RANGE_H

// RollingRange: rolling max - min of a single stream.
//
// Algorithmically identical to RollingMinMax (two monotonic deques)
// followed by a subtract; we hold the two deques directly and return
// the difference, saving the tuple-allocation that the 1->2 dispatcher
// of RollingMinMax does per step. Memory and compute order are the
// same as RollingMinMax: amortised O(1) per step.
//
// Validated in tests against RollingMinMax(w) -> max - min as the
// composition reference.

#include <cmath>
#include <limits>
#include "screamer/common/base.h"
#include "screamer/detail/monotonic_deque.h"

namespace screamer {

class RollingRange : public ScreamerBase {
public:
    explicit RollingRange(int window_size)
        : min_deque_(window_size), max_deque_(window_size) {}

    void reset() override {
        min_deque_.reset();
        max_deque_.reset();
    }

private:
    double process_scalar(double newValue) override {
        if (std::isnan(newValue)) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        const double current_min = min_deque_.append(newValue);
        const double current_max = max_deque_.append(newValue);
        return current_max - current_min;
    }

    detail::MinDeque min_deque_;
    detail::MaxDeque max_deque_;
};

}  // namespace screamer

#endif
