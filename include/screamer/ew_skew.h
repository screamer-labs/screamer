#ifndef SCREAMER_EW_SKEW_H
#define SCREAMER_EW_SKEW_H

#include <optional>
#include <stdexcept>
#include <cmath>
#include <limits>
#include "screamer/common/float_info.h"
#include "screamer/detail/running_moments_op.h"

namespace screamer {

    class EwSkew : public detail::RunningMomentsOp<EwSkew, 3, true, false, true> {
    public:
        explicit EwSkew(
            std::optional<double> com = std::nullopt,
            std::optional<double> span = std::nullopt,
            std::optional<double> halflife = std::nullopt,
            std::optional<double> alpha = std::nullopt)
        {
            // Same setup for alpha as in EwStd
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
            const double n_eff = sum_w * sum_w / st.w2;

            // Weighted mean of x - origin.
            const double mean = st.sums.s1 * inv_w;

            // Population (biased) central moments, weighted.
            const double mean_sq = mean * mean;
            const double t1 = 3 * mean * q2;
            const double t2 = 2 * mean * mean * mean;
            const double m2 = q2 - mean_sq;
            const double m3 = q3 - t1 + t2;
            // NaN once the series has stopped varying (detail/ew_flat.h).
            if (st.flat.flat(m2, peak_decay_)) {
                return std::numeric_limits<double>::quiet_NaN();
            }

            // Adjusted Fisher-Pearson skewness, the estimator scipy returns
            // for bias=False and pandas returns from rolling().skew():
            //
            //     G1 = sqrt(n(n-1))/(n-2) * m3 / m2^(3/2)
            //
            // g1 below is a ratio of *mean* moments, so sqrt(n(n-1))/(n-2) is
            // the factor that turns it into G1. The previous form,
            // n*g1/((n-1)(n-2)), applied a correction written for sum-based
            // quantities and so divided the answer by roughly n: on
            // exponential data (true skew 2.0) span=100 returned 0.019, and
            // the value shrank further as the window grew.
            double g1 = m3 / (m2 * std::sqrt(m2));
            double skew = std::sqrt(n_eff * (n_eff - 1.0)) / (n_eff - 2.0) * g1;
            if (n_eff <= 2.0) {
                return std::numeric_limits<double>::quiet_NaN();
            } else {
                return skew;
            }
        }

    private:
        double alpha_;
        double peak_decay_ = 1.0;
        
    };

} // namespace screamer

#endif // SCREAMER_EW_SKEW_H
