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
// 1 -> 1 over the returns series. One window of sums gives both the mean
// and the standard deviation.
// First valid output at sample index window_size - 1. Returns NaN
// when std is zero.

#include <cmath>
#include <limits>
#include <stdexcept>
#include "screamer/detail/window_moments_op.h"

namespace screamer {

class RollingSharpe : public detail::WindowMomentsOp<RollingSharpe, 2> {
public:
    RollingSharpe(int window_size, double periods_per_year = 1.0)
        : WindowMomentsOp(window_size, detail::StartPolicy::Strict, detail::StartPolicy::Strict),
          scale_(std::sqrt(periods_per_year))
    {
        if (window_size < 2) {
            throw std::invalid_argument("Window size must be at least 2.");
        }
        if (!(periods_per_year > 0.0)) {
            throw std::invalid_argument("periods_per_year must be positive.");
        }
        reset();
    }

    // The window mean and standard deviation from one set of sums about an
    // origin near the mean (detail::ShiftedWindowSums).
    inline double value(const State& st, double) const {
        if (st.count != static_cast<std::size_t>(window_size_)) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        const double n = static_cast<double>(window_size_);
        const double m = st.origin + st.s[0] / n;
        const double var = (st.s[1] - st.s[0] * st.s[0] / n) / (n - 1);
        const double s = std::sqrt(var);
        if (std::isnan(m) || std::isnan(s) || s <= 0.0) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        return scale_ * m / s;
    }

private:
    const double scale_;
};

}  // namespace screamer

#endif
