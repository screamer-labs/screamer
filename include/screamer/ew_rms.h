// rolling_mean.h
#ifndef SCREAMER_EW_RMS_H
#define SCREAMER_EW_RMS_H

#include <optional>
#include <stdexcept>
#include <cmath>
#include <limits>
#include "screamer/common/base.h"
#include "screamer/common/float_info.h"


namespace screamer {

    class EwRms : public ScreamerBase {
    public:
        explicit EwRms(
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

            reset();
        }

        void reset() override {
            sum_xx_ = 0.0;
            sum_w_ = 0.0;
        }

        double process_scalar(double newValue) override {
            if (isnan2(newValue)) {
                return std::numeric_limits<double>::quiet_NaN();
            }
            sum_xx_ *= one_minus_alpha_;
            sum_w_ *= one_minus_alpha_;

            sum_xx_ += newValue * newValue;
            sum_w_ += 1.0;

            // Compute the weighted mean
            double mean = sum_xx_ / sum_w_;
            return std::sqrt(mean);
        }

        // Array fast paths removed; scalar fallback honors the "ignore"
        // policy. See note in ew_mean.h.

    private:
        double alpha_;
        
        double one_minus_alpha_;
        double sum_xx_ = 0.0;
        double sum_w_ = 0.0;
    };

} // namespace screamer

#endif // SCREAMER_EW_RMW_H
