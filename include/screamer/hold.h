#ifndef SCREAMER_HOLD_H
#define SCREAMER_HOLD_H

// Hold: time-latch operator. On a nonzero finite input the output latches to
// that value and holds it for n bars total (the trigger bar plus n-1
// continuation bars). A new nonzero input mid-hold replaces the latched value
// and resets the counter. After n bars with no nonzero trigger the output
// returns to release. Zero is the trigger-absent sentinel: x != 0 means
// "trigger here." A NaN input is skipped per the library's "ignore" NaN
// policy: the output is NaN at that index and neither the held value nor the
// remaining counter is changed.
//
// Worked example (n=3, release=0.0):
//   input:  [0, 5, 0,  0,  0, -2,  0,  0]
//   output: [0, 5, 5,  5,  0, -2, -2, -2]
//
// O(1) per step. Two scalars of state.

#include <limits>
#include <stdexcept>
#include "screamer/common/base.h"
#include "screamer/common/float_info.h"

namespace screamer {

class Hold : public ScreamerBase {
public:
    Hold(int n, double release = 0.0)
        : n_(n), release_(release)
    {
        if (n < 1) {
            throw std::invalid_argument(
                "n must be >= 1.");
        }
        if (isinf2(release)) {
            throw std::invalid_argument(
                "release must be finite or NaN.");
        }
        reset();
    }

    void reset() override {
        remaining_ = 0;
        held_ = release_;
    }

private:
    double process_scalar(double x) override {
        if (isnan2(x)) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        if (x != 0.0) {
            held_ = x;
            remaining_ = n_ - 1;
            return x;
        }
        if (remaining_ > 0) {
            --remaining_;
            return held_;
        }
        return release_;
    }

    const int n_;
    const double release_;
    double held_ = std::numeric_limits<double>::quiet_NaN();
    int remaining_ = 0;
};

}  // namespace screamer

#endif
