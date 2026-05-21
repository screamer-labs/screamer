#ifndef SCREAMER_ROLLING_COV_H
#define SCREAMER_ROLLING_COV_H

// RollingCov: rolling sample covariance of two streams.
//
//     cov_w[t] = ( n*Sxy - Sx*Sy ) / ( n*(n-1) )
//
// where Sx, Sy, Sxy are the rolling sums over the window and n = window_size
// once warmup is complete. The (n-1) denominator gives the unbiased sample
// estimate, matching pandas Series.rolling(w).cov(other).
//
// Three detail::RollingSum buffers under the hood, O(1) per step.

#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include "screamer/common/functor_base.h"
#include "screamer/common/float_info.h"
#include "screamer/detail/rolling_sum.h"
#include "screamer/detail/start_policy.h"

namespace screamer {

class RollingCov : public FunctorBase<RollingCov, 2, 1> {
public:
    RollingCov(int window_size, const std::string& start_policy = "strict")
        : window_size_(window_size),
          start_policy_(detail::parse_start_policy(start_policy)),
          sum_x_buffer(window_size, "expanding"),
          sum_y_buffer(window_size, "expanding"),
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
        sum_xy_buffer.reset();
        n_ = (start_policy_ == detail::StartPolicy::Zero) ? window_size_ : 0;
    }

    ResultTuple call(const InputArray& inputs) override {
        const double x = inputs[0];
        const double y = inputs[1];

        if (isnan2(x) || isnan2(y)) {
            return std::numeric_limits<double>::quiet_NaN();
        }

        const double sum_x  = sum_x_buffer .append(x);
        const double sum_y  = sum_y_buffer .append(y);
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
        return (nd * sum_xy - sum_x * sum_y) / (nd * (nd - 1.0));
    }

private:
    const int window_size_;
    const detail::StartPolicy start_policy_;
    detail::RollingSum sum_x_buffer;
    detail::RollingSum sum_y_buffer;
    detail::RollingSum sum_xy_buffer;
    size_t n_ = 0;
};

}  // namespace screamer

#endif  // SCREAMER_ROLLING_COV_H
