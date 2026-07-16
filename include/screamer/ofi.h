#ifndef SCREAMER_OFI_H
#define SCREAMER_OFI_H

#include <limits>
#include "screamer/common/functor_base.h"
#include "screamer/common/float_info.h"

namespace screamer {

    // Order-flow imbalance: the normalized net of buy- and sell-aggressor volume,
    //     ofi = (buy - sell) / (buy + sell),
    // in [-1, 1]. Cont-Kukanov-Stoikov style signed flow. An empty bucket
    // (buy + sell == 0) has no imbalance and returns 0. Stateless and
    // elementwise; a NaN on either input yields NaN (nan_policy: ignore).
    class OFI : public FunctorBase<OFI, 2, 1> {
    public:
        OFI() = default;

        ResultTuple call(const InputArray& inputs) override {
            const double buy = inputs[0];
            const double sell = inputs[1];
            if (isnan2(buy) || isnan2(sell)) {
                return std::numeric_limits<double>::quiet_NaN();
            }
            const double total = buy + sell;
            return (total == 0.0) ? 0.0 : (buy - sell) / total;
        }
    };

} // namespace screamer

#endif // SCREAMER_OFI_H
