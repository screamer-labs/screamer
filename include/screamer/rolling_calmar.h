#ifndef SCREAMER_ROLLING_CALMAR_H
#define SCREAMER_ROLLING_CALMAR_H

// RollingCalmar: rolling Calmar ratio.
//
//     Calmar[t] = periods_per_year * mean(returns, w) /
//                 |rolling_max_drawdown(prices, w)|
//
// IMPORTANT: this class is meant to be fed RETURNS (e.g. log returns
// or simple returns). Internally it reconstructs the price path as a
// cumulative product `prod(1 + r)` so that the drawdown calculation
// makes sense. If you have the price path directly, build Calmar via
//
//     mean = RollingMean(window)(returns) * periods_per_year
//     rmd  = RollingMaxDrawdown(window)(price)
//     calmar = mean / abs(rmd)
//
// 1 -> 1. Composes RollingMean over the returns and a "rolling
// max drawdown of the implied price path".

#include <cmath>
#include <limits>
#include <stdexcept>
#include "screamer/common/base.h"
#include "screamer/rolling_max_drawdown.h"
#include "screamer/rolling_mean.h"

namespace screamer {

class RollingCalmar : public ScreamerBase {
public:
    RollingCalmar(int window_size, double periods_per_year = 1.0)
        : window_size_(window_size),
          periods_per_year_(periods_per_year),
          mean_(window_size),
          rmd_(window_size),
          price_(1.0)
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
        rmd_.reset();
        price_ = 1.0;
    }

    double process_scalar(double r) override {
        // NaN policy "ignore": leave the implied price and the wrapped
        // mean/rmd states alone.
        if (std::isnan(r)) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        // Reconstruct implied price path: price *= (1 + r).
        price_ *= (1.0 + r);
        const double m = mean_.process_scalar(r);
        const double rmd = rmd_.process_scalar(price_);
        if (std::isnan(m) || std::isnan(rmd) || rmd >= 0.0) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        return periods_per_year_ * m / std::abs(rmd);
    }

private:
    const int window_size_;
    const double periods_per_year_;
    RollingMean mean_;
    RollingMaxDrawdown rmd_;
    double price_;
};

}  // namespace screamer

#endif
