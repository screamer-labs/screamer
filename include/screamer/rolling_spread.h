#ifndef SCREAMER_ROLLING_SPREAD_H
#define SCREAMER_ROLLING_SPREAD_H

// RollingSpread: hedge-adjusted residual of x against y.
//
//     beta_w[t]   = cov(x, y) / var(y)        (same as RollingBeta)
//     spread_w[t] = x[t] - beta_w[t] * y[t]
//
// This is the rolling residual of x after regressing on y, the building
// block for pairs trading. Common usage: RollingSpread(price_a, price_b)
// where the residual measures deviation from the long-run hedge ratio.
//
// Convention: first argument is the target, second is the hedge. Matches
// RollingBeta's argument order so the same beta applies.
//
// Four detail::RollingSum buffers under the hood, O(1) per step.

#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include "screamer/common/functor_base.h"
#include "screamer/detail/rolling_sum.h"
#include "screamer/detail/start_policy.h"

namespace screamer {

class RollingSpread : public FunctorBase<RollingSpread, 2, 1> {
public:
    RollingSpread(int window_size, const std::string& start_policy = "strict")
        : window_size_(window_size),
          start_policy_(detail::parse_start_policy(start_policy)),
          sum_x_buffer(window_size, "expanding"),
          sum_y_buffer(window_size, "expanding"),
          sum_yy_buffer(window_size, "expanding"),
          sum_xy_buffer(window_size, "expanding")
    {
        if (window_size < 2) {
            throw std::invalid_argument("Window size must be 2 or more.");
        }
        reset();
    }

    void reset() override {
        sum_x_buffer.reset();
        sum_y_buffer.reset();
        sum_yy_buffer.reset();
        sum_xy_buffer.reset();
        n_ = (start_policy_ == detail::StartPolicy::Zero) ? window_size_ : 0;
    }

    ResultTuple call(const InputArray& inputs) override {
        const double x = inputs[0];
        const double y = inputs[1];

        const double sum_x  = sum_x_buffer .append(x);
        const double sum_y  = sum_y_buffer .append(y);
        const double sum_yy = sum_yy_buffer.append(y * y);
        const double sum_xy = sum_xy_buffer.append(x * y);

        if ((n_ < window_size_) && (start_policy_ != detail::StartPolicy::Zero)) {
            n_++;
        }

        if (start_policy_ == detail::StartPolicy::Strict && n_ < window_size_) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        if (n_ < 2) {
            return std::numeric_limits<double>::quiet_NaN();
        }

        const double nd = static_cast<double>(n_);
        const double num   = nd * sum_xy - sum_x * sum_y;
        const double denom = nd * sum_yy - sum_y * sum_y;

        if (denom <= 0.0) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        const double beta = num / denom;
        return x - beta * y;
    }

private:
    const int window_size_;
    const detail::StartPolicy start_policy_;
    detail::RollingSum sum_x_buffer;
    detail::RollingSum sum_y_buffer;
    detail::RollingSum sum_yy_buffer;
    detail::RollingSum sum_xy_buffer;
    size_t n_ = 0;
};

}  // namespace screamer

#endif  // SCREAMER_ROLLING_SPREAD_H
