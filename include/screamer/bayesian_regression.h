#ifndef SCREAMER_BAYESIAN_REGRESSION_H
#define SCREAMER_BAYESIAN_REGRESSION_H

// BayesianRegression: online Bayesian simple regression y = b0 + b1*x + eps,
// eps ~ N(0, sigma^2) with sigma^2 unknown. Maintains a conjugate
// Normal-Inverse-Gamma posterior over (intercept, slope, sigma^2) and updates it
// recursively with an exponential forgetting factor (stabilized forgetting: each
// step relaxes the posterior toward the prior by (1 - lambda), then folds in the new
// observation). Emits a causal one-step-ahead Student-t predictive for y_t from x_t
// using data before t, then updates. 2 -> 4:
//   [slope, intercept, pred_mean, pred_std].
// slope / intercept are the posterior means after folding in t (the current model);
// pred_mean / pred_std use data before t (the forecast). O(1) per step (2x2).
// nan_policy: ignore - a NaN x or y emits an all-NaN row and leaves state untouched.

#include <cmath>
#include <limits>
#include <optional>
#include <stdexcept>
#include <tuple>
#include "screamer/common/functor_base.h"
#include "screamer/common/float_info.h"

namespace screamer {

class BayesianRegression : public FunctorBase<BayesianRegression, 2, 4> {
public:
    BayesianRegression(
        std::optional<double> com = std::nullopt,
        std::optional<double> span = std::nullopt,
        std::optional<double> halflife = std::nullopt,
        std::optional<double> alpha = std::nullopt,
        double prior_precision = 1.0,
        double prior_sigma = 1.0)
        : prior_precision_(prior_precision), prior_sigma_(prior_sigma)
    {
        const int provided = (com.has_value() ? 1 : 0) + (span.has_value() ? 1 : 0)
                           + (halflife.has_value() ? 1 : 0) + (alpha.has_value() ? 1 : 0);
        if (provided != 1)
            throw std::invalid_argument(
                "Exactly one of com, span, halflife, or alpha must be provided");
        double a;
        if (alpha.has_value())        a = alpha.value();
        else if (com.has_value())     a = 1.0 / (1.0 + com.value());
        else if (span.has_value())    a = 2.0 / (span.value() + 1.0);
        else                          a = 1.0 - std::exp(-std::log(2.0) / halflife.value());
        if (!std::isfinite(a) || a <= 0.0 || a >= 1.0)
            throw std::invalid_argument("Alpha must be a finite value between 0 and 1 (exclusive)");
        if (prior_precision_ <= 0.0)
            throw std::invalid_argument("prior_precision must be positive");
        if (prior_sigma_ <= 0.0)
            throw std::invalid_argument("prior_sigma must be positive");
        lambda_ = 1.0 - a;                             // forgetting factor
        a0_ = 2.0;                                     // weak proper IG shape (predictive defined from step 1)
        b0_ = prior_sigma_ * prior_sigma_ * (a0_ - 1.0);   // prior E[sigma^2] = prior_sigma^2
        L0_11_ = prior_precision_; L0_12_ = 0.0; L0_22_ = prior_precision_;   // Lambda0 = prior_precision * I
        eta0_1_ = 0.0; eta0_2_ = 0.0;                  // eta0 = Lambda0 * m0, m0 = 0
        s0_ = 2.0 * b0_;                               // s0 = 2 b0 + m0^T Lambda0 m0 = 2 b0
        reset();
    }

    void reset() override {
        L11_ = L0_11_; L12_ = L0_12_; L22_ = L0_22_;
        eta1_ = eta0_1_; eta2_ = eta0_2_;
        s_ = s0_; a_ = a0_;
    }

    ResultTuple call(const InputArray& inputs) override {
        const double y = inputs[0];
        const double x = inputs[1];
        const double nan = std::numeric_limits<double>::quiet_NaN();
        if (isnan2(y) || isnan2(x))
            return std::make_tuple(nan, nan, nan, nan);  // ignore: state untouched

        const double one_minus = 1.0 - lambda_;

        // 1) stabilized forgetting: relax the posterior toward the prior.
        L11_ = lambda_ * L11_ + one_minus * L0_11_;
        L12_ = lambda_ * L12_ + one_minus * L0_12_;
        L22_ = lambda_ * L22_ + one_minus * L0_22_;
        eta1_ = lambda_ * eta1_ + one_minus * eta0_1_;
        eta2_ = lambda_ * eta2_ + one_minus * eta0_2_;
        s_   = lambda_ * s_   + one_minus * s0_;
        a_   = lambda_ * a_   + one_minus * a0_;

        // 2) one-step-ahead predictive for y from phi = [1, x], using state before t.
        double det = L11_ * L22_ - L12_ * L12_;
        double m1 = ( L22_ * eta1_ - L12_ * eta2_) / det;   // intercept mean
        double m2 = (-L12_ * eta1_ + L11_ * eta2_) / det;   // slope mean
        double pred_mean = m1 + m2 * x;
        double etaLinv_eta = (L22_ * eta1_ * eta1_ - 2.0 * L12_ * eta1_ * eta2_
                              + L11_ * eta2_ * eta2_) / det;
        double b = 0.5 * (s_ - etaLinv_eta);
        double phiLinv_phi = (L22_ - 2.0 * L12_ * x + L11_ * x * x) / det;
        double pred_var = (b / a_) * (1.0 + phiLinv_phi) * a_ / (a_ - 1.0);
        double pred_std = std::sqrt(pred_var);

        // 3) update with (phi, y): Lambda += phi phi^T, eta += phi y, s += y^2, a += 1/2.
        L11_ += 1.0;
        L12_ += x;
        L22_ += x * x;
        eta1_ += y;
        eta2_ += x * y;
        s_   += y * y;
        a_   += 0.5;

        // 4) posterior mean after the update: the reported model.
        det = L11_ * L22_ - L12_ * L12_;
        double slope     = (-L12_ * eta1_ + L11_ * eta2_) / det;
        double intercept = ( L22_ * eta1_ - L12_ * eta2_) / det;
        return std::make_tuple(slope, intercept, pred_mean, pred_std);
    }

private:
    double prior_precision_, prior_sigma_;
    double lambda_{};
    double a0_{}, b0_{}, s0_{};
    double L0_11_{}, L0_12_{}, L0_22_{};
    double eta0_1_{}, eta0_2_{};
    double L11_{}, L12_{}, L22_{};
    double eta1_{}, eta2_{};
    double s_{}, a_{};
};

}  // namespace screamer

#endif  // SCREAMER_BAYESIAN_REGRESSION_H
