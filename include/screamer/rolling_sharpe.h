#ifndef SCREAMER_ROLLING_SHARPE_H
#define SCREAMER_ROLLING_SHARPE_H

// RollingSharpe: rolling Sharpe ratio of a returns series, scaled
// to an annualisation period.
//
//     Sharpe[t] = sqrt(periods_per_year) * mean(returns) / std(returns)
//
// Both `mean` and `std` are rolling over the last `window_size`
// samples; `std` uses ddof=1 (sample std, pandas default). For
// daily returns and a 252-trading-day year use periods_per_year=252.
//
// 1 -> 1 over the returns series. Composes RollingMean + RollingStd.
// First valid output at sample index window_size - 1. Returns NaN
// when std is zero.

#include <cmath>
#include <limits>
#include <stdexcept>
#include "screamer/common/base.h"
#include "screamer/rolling_mean.h"
#include "screamer/rolling_std.h"

namespace screamer {

class RollingSharpe : public ScreamerBase {
public:
    RollingSharpe(int window_size, double periods_per_year = 1.0)
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

    double process_scalar(double r) override {
        const double m = mean_.process_scalar(r);
        const double s = std_.process_scalar(r);
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
