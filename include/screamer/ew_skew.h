#ifndef SCREAMER_EW_SKEW_H
#define SCREAMER_EW_SKEW_H

#include <optional>
#include <stdexcept>
#include <cmath>
#include <limits>
#include "screamer/common/base.h"
#include "screamer/common/float_info.h"

namespace screamer {

    class EwSkew : public ScreamerBase {
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

            reset();
        }

        void reset() override {
            sum_x_ = 0.0;
            sum_xx_ = 0.0;
            sum_xxx_ = 0.0;
            sum_w_ = 0.0;
            sum_w2_ = 0.0;
        }

        double process_scalar(double newValue) override {
            if (isnan2(newValue)) {
                return std::numeric_limits<double>::quiet_NaN();
            }
            sum_x_ *= one_minus_alpha_;
            sum_xx_ *= one_minus_alpha_;
            sum_xxx_ *= one_minus_alpha_;

            sum_w_ *= one_minus_alpha_;
            sum_w2_ *= one_minus_alpha2_;

            sum_x_ += newValue;
            sum_xx_ += newValue * newValue;
            sum_xxx_ += newValue * newValue * newValue;

            sum_w_ += 1.0;
            sum_w2_ += 1.0;

            double n_eff = sum_w_ * sum_w_ / sum_w2_;

            // Compute the weighted mean
            double mean = sum_x_ / sum_w_;

            // Population (biased) central moments, weighted.
            double m2 = (sum_xx_ / sum_w_) - (mean * mean);
            double m3 = (sum_xxx_ / sum_w_) - 3 * mean * (sum_xx_ / sum_w_) + 2 * mean * mean * mean;

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
        
        double one_minus_alpha_;
        double one_minus_alpha2_;
        double sum_x_ = 0.0;
        double sum_xx_ = 0.0;
        double sum_xxx_ = 0.0;
        double sum_w_ = 0.0;
        double sum_w2_ = 0.0;
    };

} // namespace screamer

#endif // SCREAMER_EW_SKEW_H
