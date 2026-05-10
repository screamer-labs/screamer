#ifndef SCREAMER_DIFF2_H
#define SCREAMER_DIFF2_H

// Diff2: second-order finite difference (discrete second derivative).
//
//     y[t] = x[t] - 2*x[t-1] + x[t-2]
//          = (x[t] - x[t-1]) - (x[t-1] - x[t-2])
//
// Implemented as two chained Diff(1) buffers, so the warmup naturally
// emits NaN for the first two samples under start_policy="strict".
// This is *not* the same as Diff(2), which is the lag-2 first
// difference x[t] - x[t-2].

#include <string>
#include "screamer/common/base.h"
#include "screamer/detail/delay_buffer.h"

namespace screamer {

class Diff2 : public ScreamerBase {
public:
    Diff2(const std::string& start_policy = "strict")
        : first_(1, start_policy), second_(1, start_policy)
    {}

    void reset() override {
        first_.reset();
        second_.reset();
    }

    double process_scalar(double x) override {
        const double d1 = x - first_.append(x);
        return d1 - second_.append(d1);
    }

private:
    detail::DelayBuffer first_;
    detail::DelayBuffer second_;
};

}  // namespace screamer

#endif  // SCREAMER_DIFF2_H
