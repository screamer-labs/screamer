#ifndef SCREAMER_COMMON_MARKET_PRICE_H
#define SCREAMER_COMMON_MARKET_PRICE_H

#include <limits>
#include "screamer/common/float_info.h"

namespace screamer {

    // A market order is a limit at the maximally aggressive price. A NaN price is
    // the side-agnostic market shorthand: it becomes +inf on a buy (clears any
    // offer) and -inf on a sell (hits any bid). Finite and +/-inf prices pass
    // through unchanged, so the normal fill comparison turns a +inf buy into a
    // market fill and a -inf buy into a never-fill limit.
    inline double market_limit(double price, bool buy) {
        if (isnan2(price)) {
            return buy ? std::numeric_limits<double>::infinity()
                       : -std::numeric_limits<double>::infinity();
        }
        return price;
    }

} // namespace screamer

#endif // SCREAMER_COMMON_MARKET_PRICE_H
