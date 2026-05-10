#ifndef SCREAMER_ROLLING_IQR_H
#define SCREAMER_ROLLING_IQR_H

// RollingIqr: rolling inter-quartile range = q75 - q25.
//
// Algorithm: a single OrderStatisticTree (the same primitive
// RollingQuantile uses), queried twice per step for the two quartile
// positions. O(log W) per step.
//
// Why not just compose two RollingQuantile instances? Two trees
// instead of one: 2x memory, 2x insert/erase work per step. Same
// asymptotic complexity, but a real 2x constant-factor saving.
// Validated in tests against
//
//     RollingQuantile(w, 0.75)(x) - RollingQuantile(w, 0.25)(x)
//
// as the composition reference.

#include <cmath>
#include <limits>
#include <stdexcept>
#include "screamer/common/base.h"
#include "screamer/common/buffer.h"
#include "screamer/common/order_statistic_tree.h"
#include "screamer/common/float_info.h"

namespace screamer {

class RollingIqr : public ScreamerBase {
public:
    explicit RollingIqr(int window_size)
        : window_size_(window_size),
          buffer_(window_size, std::numeric_limits<double>::quiet_NaN()),
          ost_(window_size)
    {
        if (window_size <= 0) {
            throw std::invalid_argument("Window size must be positive.");
        }
    }

    void reset() override {
        buffer_.reset(std::numeric_limits<double>::quiet_NaN());
        ost_.clear();
    }

private:
    double process_scalar(double newValue) override {
        const double oldValue = buffer_.append(newValue);
        if (!isnan2(oldValue)) {
            ost_.erase(oldValue);
        }
        if (!isnan2(newValue)) {
            ost_.insert(newValue);
        }

        if (ost_.size() < window_size_) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        return interpolated_quantile(0.75) - interpolated_quantile(0.25);
    }

    // Linear-interpolation quantile, matching RollingQuantile.
    double interpolated_quantile(double q) const {
        const int n = ost_.size();
        const double pos = q * (n - 1);
        const int idx = static_cast<int>(std::floor(pos));
        const double frac = pos - idx;
        const double lower = ost_.kth_element(idx);
        if (frac == 0.0 || idx + 1 >= n) {
            return lower;
        }
        const double upper = ost_.kth_element(idx + 1);
        return lower + frac * (upper - lower);
    }

    const int window_size_;
    FixedSizeBuffer buffer_;
    OrderStatisticTree ost_;
};

}  // namespace screamer

#endif
