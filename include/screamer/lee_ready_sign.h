#ifndef SCREAMER_LEE_READY_SIGN_H
#define SCREAMER_LEE_READY_SIGN_H

#include <limits>
#include "screamer/common/functor_base.h"
#include "screamer/common/float_info.h"

namespace screamer {

    // Lee-Ready (1991) trade sign: +1 when the trade prints above the mid, -1
    // below, and the tick-rule sign of the price when it prints exactly at the
    // mid. The tick-rule fallback state advances on every price, so it stays
    // consistent whether the mid path or the fallback is taken. A NaN on either
    // input yields NaN (nan_policy: ignore) and leaves the state untouched.
    class LeeReadySign : public FunctorBase<LeeReadySign, 2, 1> {
    public:
        LeeReadySign() { reset(); }

        void reset() override {
            have_prev_ = false;
            prev_price_ = 0.0;
            last_sign_ = std::numeric_limits<double>::quiet_NaN();
        }

        ResultTuple call(const InputArray& inputs) override {
            const double price = inputs[0];
            const double mid = inputs[1];
            if (isnan2(price) || isnan2(mid)) {
                return std::numeric_limits<double>::quiet_NaN();
            }
            // Advance the tick-rule fallback on every finite price.
            double tick;
            if (!have_prev_) {
                have_prev_ = true;
                prev_price_ = price;
                tick = std::numeric_limits<double>::quiet_NaN();
            } else {
                const double diff = price - prev_price_;
                prev_price_ = price;
                if (diff > 0.0) {
                    last_sign_ = 1.0;
                } else if (diff < 0.0) {
                    last_sign_ = -1.0;
                }
                tick = last_sign_;
            }
            if (price > mid) return 1.0;
            if (price < mid) return -1.0;
            return tick;                               // at the mid: tick-rule fallback
        }

    private:
        bool have_prev_;
        double prev_price_;
        double last_sign_;
    };

} // namespace screamer

#endif // SCREAMER_LEE_READY_SIGN_H
