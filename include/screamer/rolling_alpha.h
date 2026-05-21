#ifndef SCREAMER_ROLLING_ALPHA_H
#define SCREAMER_ROLLING_ALPHA_H

// RollingAlpha: rolling regression intercept.
//
//     alpha = mean(target) - beta * mean(regressor)
//
// Companion to RollingBeta. Same convention: the FIRST argument is
// the dependent (target), the SECOND is the regressor. Composes
// RollingBeta + RollingMean(target) + RollingMean(regressor).
// O(1) per step.

#include <limits>
#include <stdexcept>
#include <string>
#include "screamer/common/float_info.h"
#include "screamer/common/functor_base.h"
#include "screamer/rolling_beta.h"
#include "screamer/rolling_mean.h"

namespace screamer {

class RollingAlpha : public FunctorBase<RollingAlpha, 2, 1> {
public:
    RollingAlpha(int window_size, const std::string& start_policy = "strict")
        : beta_(window_size, start_policy),
          mean_x_(window_size, start_policy),
          mean_y_(window_size, start_policy)
    {
        if (window_size < 2) {
            throw std::invalid_argument("Window size must be at least 2.");
        }
    }

    void reset() override {
        beta_.reset();
        mean_x_.reset();
        mean_y_.reset();
    }

    ResultTuple call(const InputArray& inputs) override {
        const double target    = inputs[0];
        const double regressor = inputs[1];
        // NaN policy "ignore": if either input is NaN, skip the whole step
        // so all three sub-objects (beta_, mean_x_, mean_y_) stay in sync.
        if (isnan2(target) || isnan2(regressor)) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        const double beta      = beta_.call(InputArray{target, regressor});
        const double my        = mean_x_.process_scalar(target);
        const double mr        = mean_y_.process_scalar(regressor);
        if (isnan2(beta) || isnan2(my) || isnan2(mr)) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        return my - beta * mr;
    }

private:
    RollingBeta beta_;
    RollingMean mean_x_;
    RollingMean mean_y_;
};

}  // namespace screamer

#endif
