#ifndef SCREAMER_SCHMITT_TRIGGER_H
#define SCREAMER_SCHMITT_TRIGGER_H

// SchmittTrigger: hysteresis comparator (Otto Schmitt, 1934). Emits a
// binary output (1.0 / 0.0) that latches high when the input rises
// above ``upper`` and low when the input falls below ``lower``. While
// the input sits in the dead band ``[lower, upper]`` the output
// retains its previous value -- the hysteresis that gives the
// original circuit its noise immunity.
//
//   input > upper          ->  1.0  (high, latched)
//   input < lower          ->  0.0  (low,  latched)
//   lower <= input <= upper ->  previous output (no change)
//
// Until the first sample crosses either threshold the output is NaN
// because the trigger has no prior state to latch. Subsequent NaN
// inputs are skipped per the library's "ignore" NaN policy: output is
// NaN at that index and the latched state is left untouched.
//
// O(1) per step. One scalar of state.

#include <limits>
#include <stdexcept>
#include "screamer/common/base.h"
#include "screamer/common/float_info.h"

namespace screamer {

class SchmittTrigger : public ScreamerBase {
public:
    SchmittTrigger(double lower, double upper)
        : lower_(lower), upper_(upper)
    {
        if (!(lower < upper)) {
            throw std::invalid_argument(
                "lower must be strictly less than upper.");
        }
        if (isnan2(lower) || isnan2(upper)) {
            throw std::invalid_argument("lower and upper must be finite.");
        }
        reset();
    }

    void reset() override {
        output_ = std::numeric_limits<double>::quiet_NaN();
    }

private:
    double process_scalar(double x) override {
        if (isnan2(x)) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        if (x > upper_) {
            output_ = 1.0;
        } else if (x < lower_) {
            output_ = 0.0;
        }
        return output_;
    }

    const double lower_;
    const double upper_;
    double output_ = std::numeric_limits<double>::quiet_NaN();
};

}  // namespace screamer

#endif
