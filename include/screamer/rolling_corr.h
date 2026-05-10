#ifndef SCREAMER_ROLLING_CORR_H
#define SCREAMER_ROLLING_CORR_H

// RollingCorr: rolling Pearson correlation of two streams.
//
// Math:
//     r = ( n*Sxy - Sx*Sy ) / sqrt( (n*Sxx - Sx^2) * (n*Syy - Sy^2) )
//
// Where Sx, Sy, Sxx, Syy, Sxy are the rolling sums over the window.
// Uses 5 detail::RollingSum buffers internally so updates are O(1) per step.
//
// This is the first real algorithm built on FunctorBase<_, 2, 1> (two
// inputs, one output). It validates the multi-input dispatch path that's
// already implemented in functor_base.h's handle_input_Ni_1o family.

#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include "screamer/common/functor_base.h"
#include "screamer/detail/rolling_sum.h"
#include "screamer/detail/start_policy.h"

namespace screamer {

class RollingCorr : public FunctorBase<RollingCorr, 2, 1> {
public:
    RollingCorr(int window_size, const std::string& start_policy = "strict")
        : window_size_(window_size),
          start_policy_(detail::parse_start_policy(start_policy)),
          // Internal buffers always run "expanding" so the rolling sums are
          // valid throughout warmup; this class itself enforces strict
          // semantics via n_ when start_policy_ == Strict.
          sum_x_buffer(window_size, "expanding"),
          sum_y_buffer(window_size, "expanding"),
          sum_xx_buffer(window_size, "expanding"),
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
        sum_xx_buffer.reset();
        sum_yy_buffer.reset();
        sum_xy_buffer.reset();
        n_ = (start_policy_ == detail::StartPolicy::Zero) ? window_size_ : 0;
    }

    ResultTuple call(const InputArray& inputs) override {
        const double x = inputs[0];
        const double y = inputs[1];

        const double sum_x  = sum_x_buffer .append(x);
        const double sum_y  = sum_y_buffer .append(y);
        const double sum_xx = sum_xx_buffer.append(x * x);
        const double sum_yy = sum_yy_buffer.append(y * y);
        const double sum_xy = sum_xy_buffer.append(x * y);

        if ((n_ < window_size_) && (start_policy_ != detail::StartPolicy::Zero)) {
            n_++;
        }

        // Strict policy: NaN until the window is fully populated.
        if (start_policy_ == detail::StartPolicy::Strict && n_ < window_size_) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        // Need at least 2 samples for variance to be defined.
        if (n_ < 2) {
            return std::numeric_limits<double>::quiet_NaN();
        }

        const double cov   = static_cast<double>(n_) * sum_xy - sum_x * sum_y;
        const double var_x = static_cast<double>(n_) * sum_xx - sum_x * sum_x;
        const double var_y = static_cast<double>(n_) * sum_yy - sum_y * sum_y;
        const double denom2 = var_x * var_y;

        // Constant series (zero variance) -> correlation is undefined.
        if (denom2 <= 0.0) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        return cov / std::sqrt(denom2);
    }

private:
    const int window_size_;
    const detail::StartPolicy start_policy_;
    detail::RollingSum sum_x_buffer;
    detail::RollingSum sum_y_buffer;
    detail::RollingSum sum_xx_buffer;
    detail::RollingSum sum_yy_buffer;
    detail::RollingSum sum_xy_buffer;
    size_t n_ = 0;
};

}  // namespace screamer

#endif  // SCREAMER_ROLLING_CORR_H
