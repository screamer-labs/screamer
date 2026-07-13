#ifndef SCREAMER_ROLLING_LINEAR_REGRESSION_H
#define SCREAMER_ROLLING_LINEAR_REGRESSION_H

// RollingLinearRegression: 2 -> 4 OLS fit of (target ~ slope *
// regressor + intercept). Returns the full statistical tuple per step:
//
//     (slope, intercept, r_squared, stderr)
//
// where:
//   slope     = (n*Sxy - Sx*Sy) / (n*Sxx - Sx*Sx)
//   intercept = (Sy - slope*Sx) / n
//   r_squared = (n*Sxy - Sx*Sy)^2 / ((n*Sxx - Sx*Sx) * (n*Syy - Sy*Sy))
//   stderr    = sqrt(SSE / (n - 2)),  SSE = SST - SSR
//
// 2 -> 4 functor over the N->M dispatcher. Convention
// matches RollingBeta: FIRST argument is the dependent (target),
// SECOND is the regressor. Composes five detail::RollingSum buffers
// (Sx, Sy, Sxx, Syy, Sxy). O(1) per step.

#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <tuple>
#include "screamer/common/functor_base.h"
#include "screamer/common/float_info.h"
#include "screamer/detail/rolling_sum.h"
#include "screamer/detail/start_policy.h"

namespace screamer {

class RollingLinearRegression : public FunctorBase<RollingLinearRegression, 2, 4> {
public:
    RollingLinearRegression(int window_size,
                            const std::string& start_policy = "strict")
        : window_size_(window_size),
          start_policy_(detail::parse_start_policy(start_policy)),
          sum_x_(window_size, "expanding"),
          sum_y_(window_size, "expanding"),
          sum_xx_(window_size, "expanding"),
          sum_yy_(window_size, "expanding"),
          sum_xy_(window_size, "expanding")
    {
        if (window_size < 3) {
            throw std::invalid_argument("Window size must be at least 3 (n-2 in stderr).");
        }
    }

    void reset() override {
        sum_x_.reset();
        sum_y_.reset();
        sum_xx_.reset();
        sum_yy_.reset();
        sum_xy_.reset();
        n_ = (start_policy_ == detail::StartPolicy::Zero) ? window_size_ : 0;
    }

    ResultTuple call(const InputArray& inputs) override {
        const double target = inputs[0];  // y (dependent)
        const double regr   = inputs[1];  // x (regressor)

        if (isnan2(target) || isnan2(regr)) {
            const double nan = std::numeric_limits<double>::quiet_NaN();
            return std::make_tuple(nan, nan, nan, nan);
        }

        const double Sy  = sum_y_ .append(target);
        const double Sx  = sum_x_ .append(regr);
        const double Syy = sum_yy_.append(target * target);
        const double Sxx = sum_xx_.append(regr   * regr);
        const double Sxy = sum_xy_.append(target * regr);

        if ((n_ < window_size_) && (start_policy_ != detail::StartPolicy::Zero)) {
            n_++;
        }
        const double nan = std::numeric_limits<double>::quiet_NaN();
        if (start_policy_ == detail::StartPolicy::Strict && n_ < window_size_) {
            return std::make_tuple(nan, nan, nan, nan);
        }
        if (n_ < 3) {
            return std::make_tuple(nan, nan, nan, nan);
        }

        const double nd = static_cast<double>(n_);
        const double denom_x = nd * Sxx - Sx * Sx;
        if (denom_x <= 0.0) {
            return std::make_tuple(nan, nan, nan, nan);
        }
        const double num     = nd * Sxy - Sx * Sy;
        const double slope   = num / denom_x;
        const double intercept = (Sy - slope * Sx) / nd;

        const double denom_y = nd * Syy - Sy * Sy;
        const double r_sq    = (denom_y > 0.0)
                             ? (num * num) / (denom_x * denom_y)
                             : nan;

        // SST = Syy - Sy^2/n; SSR = slope * (Sxy - Sx*Sy/n); SSE = SST - SSR.
        const double SST = Syy - Sy * Sy / nd;
        const double SSR = slope * (Sxy - Sx * Sy / nd);
        const double SSE = SST - SSR;
        const double stderr_val = (SSE > 0.0)
                                ? std::sqrt(SSE / (nd - 2.0))
                                : 0.0;

        return std::make_tuple(slope, intercept, r_sq, stderr_val);
    }

private:
    const int window_size_;
    const detail::StartPolicy start_policy_;
    detail::RollingSum sum_x_;
    detail::RollingSum sum_y_;
    detail::RollingSum sum_xx_;
    detail::RollingSum sum_yy_;
    detail::RollingSum sum_xy_;
    int n_ = 0;
};

}  // namespace screamer

#endif
