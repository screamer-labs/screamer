#ifndef SCREAMER_EFFECTIVE_SPREAD_H
#define SCREAMER_EFFECTIVE_SPREAD_H

#include <cmath>
#include <limits>
#include "screamer/common/functor_base.h"
#include "screamer/common/float_info.h"

namespace screamer {

    // Effective spread: twice the absolute distance of the trade price from the
    // mid-quote at trade time,
    //     effective = 2 * |price - mid|.
    // It is the round-trip cost actually paid relative to the mid (a buy lifts the
    // offer, a sell hits the bid), and unlike the quoted spread it reflects where
    // trades really print. Stateless and elementwise; a NaN on either input yields
    // NaN (nan_policy: ignore). Pair it with `RealizedSpread` to split the cost
    // into a realized (liquidity) part and a price-impact (adverse-selection) part.
    class EffectiveSpread : public FunctorBase<EffectiveSpread, 2, 1> {
    public:
        EffectiveSpread() = default;

        ResultTuple call(const InputArray& inputs) override {
            const double price = inputs[0];
            const double mid = inputs[1];
            if (isnan2(price) || isnan2(mid)) {
                return std::numeric_limits<double>::quiet_NaN();
            }
            return 2.0 * std::abs(price - mid);
        }
    };

} // namespace screamer

#endif // SCREAMER_EFFECTIVE_SPREAD_H
