#ifndef SCREAMER_ROLLING_MIN_H
#define SCREAMER_ROLLING_MIN_H

#include <cmath>
#include <limits>
#include "screamer/common/base.h"
#include "screamer/detail/monotonic_deque.h"

namespace screamer {

class RollingMin : public ScreamerBase {
public:
    explicit RollingMin(int window_size) : deque_(window_size) {}

    void reset() override { deque_.reset(); }

private:
    double process_scalar(double newValue) override {
        // NaN policy "ignore": leave the deque state untouched and emit NaN.
        if (std::isnan(newValue)) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        return deque_.append(newValue);
    }

    detail::MinDeque deque_;
};

}  // namespace screamer

#endif
