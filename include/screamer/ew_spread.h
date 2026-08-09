#ifndef SCREAMER_EW_SPREAD_H
#define SCREAMER_EW_SPREAD_H

// EwSpread: exponentially-weighted hedge-adjusted residual of x against y.
//
//     spread[t] = x[t] - beta_ew[t] * y[t],   beta_ew = ewcov(x, y) / ewvar(y)
//
// The EW analog of RollingSpread, using the same beta as EwBeta. Convention
// matches EwBeta / RollingSpread: the FIRST argument is the target, the SECOND
// is the hedge/regressor. The building block for EW pairs-trading residuals.
// Composes EwBeta. O(1) per step. NaN policy "ignore": a NaN in either input
// skips the whole step so the beta stays in sync.

#include <limits>
#include <optional>
#include "screamer/common/functor_base.h"
#include "screamer/common/float_info.h"
#include "screamer/ew_beta.h"

namespace screamer {

class EwSpread : public FunctorBase<EwSpread, 2, 1> {
public:
    explicit EwSpread(
        std::optional<double> com = std::nullopt,
        std::optional<double> span = std::nullopt,
        std::optional<double> halflife = std::nullopt,
        std::optional<double> alpha = std::nullopt)
        : beta_(com, span, halflife, alpha) {}

    void reset() override { beta_.reset(); }

    ResultTuple call(const InputArray& inputs) override {
        const double x = inputs[0];
        const double y = inputs[1];
        if (isnan2(x) || isnan2(y)) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        const double beta = beta_.call(InputArray{x, y});
        if (isnan2(beta)) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        return x - beta * y;
    }

private:
    EwBeta beta_;
};

}  // namespace screamer

#endif  // SCREAMER_EW_SPREAD_H
