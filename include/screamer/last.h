#ifndef SCREAMER_LAST_H
#define SCREAMER_LAST_H

// NaN policy "ignore": a NaN input is skipped --
// output is NaN at that index, the retained last value is unchanged.

#include <cmath>
#include <limits>
#include "screamer/common/base.h"

namespace screamer {

class Last : public ScreamerBase {
public:
    Last() { reset(); }

    void reset() override {
        last_value_ = std::numeric_limits<double>::quiet_NaN();
    }

    double process_scalar(double x) override {
        if (std::isnan(x)) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        last_value_ = x;
        return last_value_;
    }

private:
    double last_value_ = std::numeric_limits<double>::quiet_NaN();
};

}  // namespace screamer
#endif
