// rolling_mean.h
#ifndef SCREAMER_EW_ZSCORE_H
#define SCREAMER_EW_ZSCORE_H

#include <optional>
#include <stdexcept>
#include <cmath>
#include <limits>
#include "screamer/common/float_info.h"
#include "screamer/detail/running_moments_op.h"


namespace screamer {

    class EwZscore : public detail::RunningMomentsOp<EwZscore, 2, true> {
    public:
        explicit EwZscore(
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

        inline double value(const State& st, double x) const {
            const double sum_w = st.sums.w;
            const double inv_w = 1.0 / sum_w;
            const double q2 = st.sums.s2 * inv_w;
            const double n_eff = sum_w * sum_w / st.w2;
            // Weighted mean of x - origin; each product is its own statement
            // (see detail/shifted_power_sums.h).
            const double mean = st.sums.s1 * inv_w;
            const double mean_sq = mean * mean;
            double variance = q2 - mean_sq;
            variance *= n_eff / (n_eff - 1.0);
            return ((x - st.sums.origin) - mean) / std::sqrt(variance);
        }

    private:
        double alpha_;
        
    };

} // namespace screamer

#endif // SCREAMER_EW_ZSCORE_H
