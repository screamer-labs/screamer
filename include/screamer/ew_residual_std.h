#ifndef SCREAMER_EW_RESIDUAL_STD_H
#define SCREAMER_EW_RESIDUAL_STD_H

// EwResidualStd: exponentially-weighted std of the spread x - beta_ew * y.
//
// The EW analog of RollingResidualStd; useful for EW pairs-trading
// normalisation -- the z-score of a spread is (spread - ewmean(spread)) /
// EwResidualStd. Convention matches EwBeta / EwSpread: the FIRST argument is
// the target, the SECOND is the regressor. Composes EwSpread + EwStd. O(1) per
// step.

#include <cmath>
#include <limits>
#include <optional>
#include "screamer/common/functor_base.h"
#include "screamer/ew_spread.h"
#include "screamer/ew_std.h"

namespace screamer {

class EwResidualStd : public FunctorBase<EwResidualStd, 2, 1> {
public:
    explicit EwResidualStd(
        std::optional<double> com = std::nullopt,
        std::optional<double> span = std::nullopt,
        std::optional<double> halflife = std::nullopt,
        std::optional<double> alpha = std::nullopt)
        : spread_(com, span, halflife, alpha),
          std_(com, span, halflife, alpha) {}

    void reset() override { spread_.reset(); std_.reset(); }

    ResultTuple call(const InputArray& inputs) override {
        // EwSpread emits NaN during warm-up; feeding NaN into EwStd's running
        // sums would poison them, so skip the feed until the spread is valid.
        const double spread_val = spread_.call(inputs);
        if (std::isnan(spread_val)) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        return std_.process_scalar(spread_val);
    }

private:
    EwSpread spread_;
    EwStd std_;
};

}  // namespace screamer

#endif  // SCREAMER_EW_RESIDUAL_STD_H
