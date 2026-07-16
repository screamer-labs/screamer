#ifndef SCREAMER_TICK_RULE_SIGN_H
#define SCREAMER_TICK_RULE_SIGN_H

#include <limits>
#include "screamer/common/base.h"
#include "screamer/common/float_info.h"

namespace screamer {

    // Trade sign by the tick rule: +1 on an up-tick, -1 on a down-tick, and the
    // previous sign carried forward when the price is unchanged. The output is
    // NaN until the first directional move (no sign to carry yet) and at the
    // first sample (no prior price). A NaN price is treated as a missing tick
    // (nan_policy: ignore): the output is NaN and the state is left untouched, so
    // the next real tick is compared against the last real price.
    class TickRuleSign : public ScreamerBase {
    public:
        TickRuleSign() { reset(); }

        void reset() override {
            have_prev_ = false;
            prev_price_ = 0.0;
            last_sign_ = std::numeric_limits<double>::quiet_NaN();
        }

        double process_scalar(double price) override {
            if (isnan2(price)) {                       // ignore: missing tick
                return std::numeric_limits<double>::quiet_NaN();
            }
            if (!have_prev_) {                         // first real tick: no prior price
                have_prev_ = true;
                prev_price_ = price;
                return std::numeric_limits<double>::quiet_NaN();
            }
            const double diff = price - prev_price_;
            prev_price_ = price;
            if (diff > 0.0) {
                last_sign_ = 1.0;
            } else if (diff < 0.0) {
                last_sign_ = -1.0;
            }
            // unchanged: carry last_sign_ (NaN until the first directional move)
            return last_sign_;
        }

    private:
        bool have_prev_;
        double prev_price_;
        double last_sign_;
    };

} // namespace screamer

#endif // SCREAMER_TICK_RULE_SIGN_H
