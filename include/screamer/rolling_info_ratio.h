#ifndef SCREAMER_ROLLING_INFO_RATIO_H
#define SCREAMER_ROLLING_INFO_RATIO_H

// RollingInfoRatio: rolling Information Ratio of a returns series
// against a benchmark.
//
//     excess[t] = returns - benchmark
//     IR[t]     = sqrt(periods_per_year) * mean(excess) / std(excess)
//
// 2 -> 1 over (returns, benchmark). Essentially RollingSharpe applied
// to the active-return series. Composes RollingMean + RollingStd over
// the per-step difference.

#include <cmath>
#include <limits>
#include <stdexcept>
#include "screamer/common/functor_base.h"
#include "screamer/rolling_mean.h"
#include "screamer/rolling_std.h"

namespace screamer {

class RollingInfoRatio : public FunctorBase<RollingInfoRatio, 2, 1> {
public:
    RollingInfoRatio(int window_size, double periods_per_year = 1.0)
        : window_size_(window_size),
          scale_(std::sqrt(periods_per_year)),
          mean_(window_size),
          std_(window_size)
    {
        if (window_size < 2) {
            throw std::invalid_argument("Window size must be at least 2.");
        }
        if (!(periods_per_year > 0.0)) {
            throw std::invalid_argument("periods_per_year must be positive.");
        }
    }

    void reset() override {
        mean_.reset();
        std_.reset();
    }

    ResultTuple call(const InputArray& inputs) override {
        // NaN policy "ignore": if either input is NaN, leave the wrapped
        // mean_/std_ alone so they stay synchronized.
        if (std::isnan(inputs[0]) || std::isnan(inputs[1])) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        const double excess = inputs[0] - inputs[1];
        const double m = mean_.process_scalar(excess);
        const double s = std_.process_scalar(excess);
        if (std::isnan(m) || std::isnan(s) || s <= 0.0) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        return scale_ * m / s;
    }

private:
    const int window_size_;
    const double scale_;
    RollingMean mean_;
    RollingStd std_;
};

}  // namespace screamer

#endif
