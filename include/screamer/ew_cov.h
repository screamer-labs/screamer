#ifndef SCREAMER_EW_COV_H
#define SCREAMER_EW_COV_H

// EwCov: exponentially-weighted bias-corrected covariance of two streams.
//
//     cov[t] = ( sum_xy / sum_w - mean_x * mean_y ) * n_eff / (n_eff - 1)
//
// Same bias-correction convention as EwVar: n_eff = sum_w^2 / sum_w2,
// matching pandas.Series.ewm(adjust=True, bias=False).cov(other).
// O(1) per update.

#include <cmath>
#include <limits>
#include <optional>
#include <stdexcept>
#include "screamer/common/functor_base.h"
#include "screamer/common/float_info.h"
#include "screamer/detail/shifted_power_sums.h"

namespace screamer {

class EwCov : public FunctorBase<EwCov, 2, 1> {
public:
    // Measured faster with its step kept out of the batch column loop.
    // The column loop runs on a local copy, its state in registers.
    static constexpr bool kColumnCopy = true;

    explicit EwCov(
        std::optional<double> com = std::nullopt,
        std::optional<double> span = std::nullopt,
        std::optional<double> halflife = std::nullopt,
        std::optional<double> alpha = std::nullopt)
    {
        const int provided = (com.has_value() ? 1 : 0)
                           + (span.has_value() ? 1 : 0)
                           + (halflife.has_value() ? 1 : 0)
                           + (alpha.has_value() ? 1 : 0);
        if (provided != 1) {
            throw std::invalid_argument("Exactly one of com, span, halflife, or alpha must be provided");
        }
        if (alpha.has_value())          alpha_ = alpha.value();
        else if (com.has_value())       alpha_ = 1.0 / (1.0 + com.value());
        else if (span.has_value())      alpha_ = 2.0 / (span.value() + 1.0);
        else                            alpha_ = 1.0 - std::exp(-std::log(2.0) / halflife.value());

        if (!std::isfinite(alpha_) || alpha_ <= 0.0 || alpha_ >= 1.0) {
            throw std::invalid_argument("Alpha must be a finite value between 0 and 1 (exclusive)");
        }
        one_minus_alpha_  = 1.0 - alpha_;
        one_minus_alpha2_ = one_minus_alpha_ * one_minus_alpha_;
        reset();
    }

    void reset() override {
        sums_.reset();
        sum_w2_ = 0.0;
    }

    ResultTuple call(const InputArray& inputs) override {
        const double x = inputs[0];
        const double y = inputs[1];

        if (isnan2(x) || isnan2(y)) {
            return std::numeric_limits<double>::quiet_NaN();
        }

        // Decay every weight by (1 - alpha), then add the pair with weight 1.
        // The sums are taken about origins that follow the means, so they do
        // not cancel when the levels are large next to the spread; the
        // formulas below do not depend on the origins.
        sums_.decay(one_minus_alpha_);
        sum_w2_ *= one_minus_alpha2_;
        sums_.add(x, y);
        sum_w2_ += 1.0;

        const double sum_w = sums_.w;
        const double inv_w = 1.0 / sum_w;
        const double q_xy = sums_.sxy * inv_w;
        const double n_eff = sum_w * sum_w / sum_w2_;
        if (n_eff <= 1.0) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        // Each product is its own statement (see detail/shifted_power_sums.h).
        const double mean_x = sums_.sx * inv_w;
        const double mean_y = sums_.sy * inv_w;
        const double mean_xy = mean_x * mean_y;
        const double cov_biased = q_xy - mean_xy;
        return cov_biased * n_eff / (n_eff - 1.0);
    }

private:
    double alpha_{};
    double one_minus_alpha_{};
    double one_minus_alpha2_{};
    detail::ShiftedCoPowerSums sums_;
    double sum_w2_ = 0.0;
};

}  // namespace screamer

#endif  // SCREAMER_EW_COV_H
