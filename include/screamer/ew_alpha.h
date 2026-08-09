#ifndef SCREAMER_EW_ALPHA_H
#define SCREAMER_EW_ALPHA_H

// EwAlpha: exponentially-weighted regression intercept.
//
//     alpha[t] = ewmean(target) - beta_ew * ewmean(regressor)
//
// The EW analog of RollingAlpha, companion to EwBeta. Convention matches
// EwBeta: the FIRST argument is the target, the SECOND is the regressor.
// Composes EwBeta + EwMean(target) + EwMean(regressor). O(1) per step. NaN
// policy "ignore": a NaN in either input skips the whole step so the three
// sub-objects stay in sync.

#include <limits>
#include <optional>
#include "screamer/common/functor_base.h"
#include "screamer/common/float_info.h"
#include "screamer/ew_beta.h"
#include "screamer/ew_mean.h"

namespace screamer {

class EwAlpha : public FunctorBase<EwAlpha, 2, 1> {
public:
    explicit EwAlpha(
        std::optional<double> com = std::nullopt,
        std::optional<double> span = std::nullopt,
        std::optional<double> halflife = std::nullopt,
        std::optional<double> alpha = std::nullopt)
        : beta_(com, span, halflife, alpha),
          mean_x_(com, span, halflife, alpha),
          mean_y_(com, span, halflife, alpha) {}

    void reset() override { beta_.reset(); mean_x_.reset(); mean_y_.reset(); }

    ResultTuple call(const InputArray& inputs) override {
        const double target    = inputs[0];
        const double regressor = inputs[1];
        if (isnan2(target) || isnan2(regressor)) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        const double beta = beta_.call(InputArray{target, regressor});
        const double my   = mean_x_.process_scalar(target);
        const double mr   = mean_y_.process_scalar(regressor);
        if (isnan2(beta) || isnan2(my) || isnan2(mr)) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        return my - beta * mr;
    }

private:
    EwBeta beta_;
    EwMean mean_x_;
    EwMean mean_y_;
};

}  // namespace screamer

#endif  // SCREAMER_EW_ALPHA_H
