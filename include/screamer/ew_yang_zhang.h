#ifndef SCREAMER_EW_YANG_ZHANG_H
#define SCREAMER_EW_YANG_ZHANG_H

// Exponentially-weighted Yang-Zhang volatility (Yang & Zhang, 2000), the EW
// analog of RollingYangZhang and the completion of the EW range-based
// volatility family (EwParkinson, EwGarmanKlass, EwRogersSatchell).
//
//     sigma2_o   = EW variance of overnight log returns ln(O[t] / C[t-1])
//     sigma2_c   = EW variance of open-to-close log returns ln(C[t] / O[t])
//     sigma2_rs  = EW mean of the per-bar Rogers-Satchell estimate
//     k          = 0.34 / (1.34 + (n_eff+1)/(n_eff-1))
//     sigma2_YZ  = sigma2_o + k * sigma2_c + (1-k) * sigma2_rs
//
// 4 -> 1 over (open, high, low, close). The rolling version fixes k from the
// window size n; the EW version uses the effective sample size n_eff of the
// overnight component, so k adapts as the weighting fills. Composition: two
// EwVar (overnight + open-to-close) and one EwMean (Rogers-Satchell per bar).
// The first overnight return is undefined (no prior close), so the output is
// NaN until at least two overnight returns exist. O(1) per step. NaN policy
// "ignore": a bar with any missing field is skipped whole.

#include <cmath>
#include <limits>
#include <optional>
#include <stdexcept>
#include "screamer/common/float_info.h"
#include "screamer/common/functor_base.h"
#include "screamer/ew_var.h"
#include "screamer/ew_mean.h"

namespace screamer {

class EwYangZhangVar : public FunctorBase<EwYangZhangVar, 4, 1> {
public:
    explicit EwYangZhangVar(
        std::optional<double> com = std::nullopt,
        std::optional<double> span = std::nullopt,
        std::optional<double> halflife = std::nullopt,
        std::optional<double> alpha = std::nullopt)
        : var_overnight_(com, span, halflife, alpha),
          var_open_close_(com, span, halflife, alpha),
          rs_mean_(com, span, halflife, alpha)
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
        var_overnight_.reset();
        var_open_close_.reset();
        rs_mean_.reset();
        prev_close_ = std::numeric_limits<double>::quiet_NaN();
        sum_w_ = sum_w2_ = 0.0;
    }

    ResultTuple call(const InputArray& inputs) override {
        const double O = inputs[0];
        const double H = inputs[1];
        const double L = inputs[2];
        const double C = inputs[3];
        if (any_nan(O, H, L, C)) {
            return std::numeric_limits<double>::quiet_NaN();
        }

        const double oc = std::log(C / O);
        const double rs = std::log(H / C) * std::log(H / O)
                        + std::log(L / C) * std::log(L / O);
        const double v_oc = var_open_close_.process_scalar(oc);
        const double v_rs = rs_mean_.process_scalar(rs);

        double v_on = std::numeric_limits<double>::quiet_NaN();
        if (!isnan2(prev_close_)) {
            const double on = std::log(O / prev_close_);
            v_on = var_overnight_.process_scalar(on);
            // Track the overnight component's effective sample size for k.
            sum_w_  = one_minus_alpha_ * sum_w_ + 1.0;
            sum_w2_ = one_minus_alpha2_ * sum_w2_ + 1.0;
        }
        prev_close_ = C;

        if (isnan2(v_on) || isnan2(v_oc) || isnan2(v_rs)) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        const double n_eff = sum_w_ * sum_w_ / sum_w2_;
        if (n_eff <= 1.0) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        const double k = 0.34 / (1.34 + (n_eff + 1.0) / (n_eff - 1.0));
        return v_on + k * v_oc + (1.0 - k) * v_rs;
    }

private:
    double alpha_{};
    double one_minus_alpha_{};
    double one_minus_alpha2_{};
    EwVar var_overnight_;
    EwVar var_open_close_;
    EwMean rs_mean_;
    double prev_close_ = std::numeric_limits<double>::quiet_NaN();
    double sum_w_ = 0.0;
    double sum_w2_ = 0.0;
};

class EwYangZhangVol : public FunctorBase<EwYangZhangVol, 4, 1> {
public:
    explicit EwYangZhangVol(
        std::optional<double> com = std::nullopt,
        std::optional<double> span = std::nullopt,
        std::optional<double> halflife = std::nullopt,
        std::optional<double> alpha = std::nullopt)
        : var_(com, span, halflife, alpha) {}
    void reset() override { var_.reset(); }
    ResultTuple call(const InputArray& inputs) override {
        return std::sqrt(var_.call(inputs));
    }
private:
    EwYangZhangVar var_;
};

}  // namespace screamer

#endif  // SCREAMER_EW_YANG_ZHANG_H
