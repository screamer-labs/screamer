#ifndef SCREAMER_EW_KURT_H
#define SCREAMER_EW_KURT_H

#include <optional>
#include <stdexcept>
#include <cmath>
#include <limits>
#include "screamer/common/float_info.h"
#include "screamer/detail/running_moments_op.h"

namespace screamer {

    class EwKurt : public detail::RunningMomentsOp<EwKurt, 4, true, false, true> {
    public:
        explicit EwKurt(
            std::optional<double> com = std::nullopt,
            std::optional<double> span = std::nullopt,
            std::optional<double> halflife = std::nullopt,
            std::optional<double> alpha = std::nullopt)
        {
            int provided_args = (com.has_value() ? 1 : 0) +
                                (span.has_value() ? 1 : 0) +
                                (halflife.has_value() ? 1 : 0) +
                                (alpha.has_value() ? 1 : 0);

            if (provided_args != 1) {
                throw std::invalid_argument("Exactly one of com, span, halflife, or alpha must be provided");
            }

            if (alpha.has_value()) {
                alpha_ = alpha.value();
            } else if (com.has_value()) {
                alpha_ = 1.0 / (1.0 + com.value());
            } else if (span.has_value()) {
                alpha_ = 2.0 / (span.value() + 1.0);
            } else if (halflife.has_value()) {
                alpha_ = 1.0 - std::exp(-std::log(2.0) / halflife.value());
            }

            if (!std::isfinite(alpha_) || alpha_ <= 0.0 || alpha_ >= 1.0) {
                throw std::invalid_argument("Alpha must be a finite value between 0 and 1 (exclusive)");
            }
            one_minus_alpha_ = 1.0 - alpha_;
            one_minus_alpha2_ = one_minus_alpha_ * one_minus_alpha_;
            peak_decay_ = detail::ew_flat_peak_decay(one_minus_alpha_);

            reset();
        }

        inline double value(State& st, double) const {
            // The power sums are taken about an origin that follows the mean,
            // so they do not cancel when the mean is large next to the spread;
            // the formulas below do not depend on the origin. Each product is
            // its own statement (see detail/shifted_power_sums.h).
            const double sum_w = st.sums.w;
            const double inv_w = 1.0 / sum_w;
            const double q2 = st.sums.s2 * inv_w;
            const double q3 = st.sums.s3 * inv_w;
            const double q4 = st.sums.s4 * inv_w;
            const double n_eff = sum_w * sum_w / st.w2;

            // Weighted mean of x - origin.
            const double mean = st.sums.s1 * inv_w;
            const double mean2 = mean * mean;

            // Population (biased) second central moment, weighted.
            const double m2 = q2 - mean2;
            // NaN once the series has stopped varying (detail/ew_flat.h).
            if (st.flat.flat(m2, peak_decay_)) {
                return std::numeric_limits<double>::quiet_NaN();
            }

            // Compute the fourth central moment (m4)
            const double t1 = 4 * mean * q3;
            const double t2 = 6 * mean2 * q2;
            const double t3 = 3 * mean2 * mean2;
            const double m4 = q4 - t1 + t2 - t3;

            // Adjusted excess kurtosis, the estimator scipy returns for
            // bias=False and pandas returns from rolling().kurt():
            //
            //     G2 = (n-1)/((n-2)(n-3)) * ((n+1) * m4/m2^2 - 3(n-1))
            //
            // g2 is a ratio of *mean* moments. The previous form divided by
            // (n-1)(n-2)(n-3) without the compensating factor, which shrank
            // the result by roughly n: it returned ~0 for every distribution
            // tested (uniform -1.2, laplace 3, exponential 6), and looked
            // correct only on normal data, where the true answer is 0.
            const double g2 = m4 / (m2 * m2);
            const double u = (n_eff + 1.0) * g2;
            const double v = 3.0 * (n_eff - 1.0);
            const double excess_kurtosis = (n_eff - 1.0) / ((n_eff - 2.0) * (n_eff - 3.0)) * (u - v);
            if (n_eff <= 3.0) {
                return std::numeric_limits<double>::quiet_NaN();
            } else {
                return excess_kurtosis;
            }
        }

    private:
        double alpha_;
        double peak_decay_ = 1.0;
        
    };

} // namespace screamer

#endif // SCREAMER_EW_KURT_H
