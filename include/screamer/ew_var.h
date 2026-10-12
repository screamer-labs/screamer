// rolling_mean.h
#ifndef SCREAMER_EW_VAR_H
#define SCREAMER_EW_VAR_H

#include <optional>
#include <stdexcept>
#include <cmath>
#include <limits>
#include "screamer/common/float_info.h"
#include "screamer/detail/running_moments_op.h"

/*
Info about the bias correction: https://osquant.com/papers/replicating-pandas-ewm-var/ 

Exponetially weighted statistics have a sum of weights that's a geometric series

Geometric series:
sum_w = 1 + a + a^2 + ... + a^n 
      = (1 - a^(n+1)) / (1 - a)

The bias correction term for exponentially weighted variance is:

b = sum_w^2 / ( sum_w^2 - sum_w2)

with 
sum_w2 = 1 + a^2 + a^4 + ... + a^2n
       = 1 - a^(2*(n+1))) / (1 - a^2)

and we should get
b = (1 - w^{n+1}) (1 + w) / [ 2w (1-w^n) ]

A simpler way is to compute "effective sample size" for exponentialy weighted 
statistical indicators, and plug that value in for N in the unbiased sample
estimates. Like for the variance estimate: sum (x - mu)^2 / (N - 1)

n_eff = S_1^2 / S_2

S_1 <- (1 - alpha) S_1 + 1
S_2 <- (1 - alpha)^2 S_2 + 1
*/

namespace screamer {

    class EwVar : public detail::RunningMomentsOp<EwVar, 2, true> {
    public:
        explicit EwVar(
            std::optional<double> com = std::nullopt,
            std::optional<double> span = std::nullopt,
            std::optional<double> halflife = std::nullopt,
            std::optional<double> alpha = std::nullopt)
        {
            // Count the number of provided arguments
            int provided_args = (com.has_value() ? 1 : 0) +
                                (span.has_value() ? 1 : 0) +
                                (halflife.has_value() ? 1 : 0) +
                                (alpha.has_value() ? 1 : 0);

            if (provided_args != 1) {
                throw std::invalid_argument("Exactly one of com, span, halflife, or alpha must be provided");
            }

            // Map provided argument to alpha
            if (alpha.has_value()) {
                alpha_ = alpha.value();
            } else if (com.has_value()) {
                alpha_ = 1.0 / (1.0 + com.value());
            } else if (span.has_value()) {
                alpha_ = 2.0 / (span.value() + 1.0);
            } else if (halflife.has_value()) {
                alpha_ = 1.0 - std::exp(-std::log(2.0) / halflife.value());
            }

            // Validate alpha
            if (!std::isfinite(alpha_) || alpha_ <= 0.0 || alpha_ >= 1.0) {
                throw std::invalid_argument("Alpha must be a finite value between 0 and 1 (exclusive)");
            }
            one_minus_alpha_ = 1.0 - alpha_;
            one_minus_alpha2_ = one_minus_alpha_*one_minus_alpha_;

            reset();
        }

        inline double value(const State& st, double) const {
            const double sum_w = st.sums.w;
            const double inv_w = 1.0 / sum_w;
            const double q2 = st.sums.s2 * inv_w;
            const double n_eff = sum_w * sum_w / st.w2;
            // Weighted mean of x - origin; the variance does not depend on the
            // origin. Each product is its own statement (see
            // detail/shifted_power_sums.h).
            const double mean = st.sums.s1 * inv_w;
            const double mean_sq = mean * mean;
            double variance = q2 - mean_sq;
            variance *= n_eff / (n_eff - 1.0);

            if (n_eff <= 1.0) {
                return std::numeric_limits<double>::quiet_NaN();
            } else {
                return variance;
            }
        }

    private:
        double alpha_;
        
    };

} // namespace screamer

#endif // SCREAMER_EW_VAR_H
