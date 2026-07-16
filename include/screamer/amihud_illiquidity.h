#ifndef SCREAMER_AMIHUD_ILLIQUIDITY_H
#define SCREAMER_AMIHUD_ILLIQUIDITY_H

#include <cmath>
#include <limits>
#include <string>
#include "screamer/common/functor_base.h"
#include "screamer/common/float_info.h"
#include "screamer/detail/rolling_mean.h"

namespace screamer {

    // Amihud (2002) illiquidity: the trailing-window mean of |return| / notional.
    // Large values mean price moves a lot per unit traded (an illiquid,
    // high-impact regime), a robust and cheap cousin of Kyle's lambda. A zero
    // notional, or a NaN on either input, contributes no sample to the window
    // and yields NaN at that step (nan_policy: ignore).
    class AmihudIlliquidity : public FunctorBase<AmihudIlliquidity, 2, 1> {
    public:
        AmihudIlliquidity(int window_size, const std::string& start_policy = "strict")
            : mean_(window_size, start_policy)
        {
        }

        void reset() override {
            mean_.reset();
        }

        ResultTuple call(const InputArray& inputs) override {
            const double ret = inputs[0];
            const double notional = inputs[1];
            double ratio;
            if (isnan2(ret) || isnan2(notional) || notional == 0.0) {
                ratio = std::numeric_limits<double>::quiet_NaN();
            } else {
                ratio = std::abs(ret) / notional;
            }
            // detail::RollingMean.append ignores a NaN ratio (state untouched,
            // returns NaN), giving the "ignore" behavior for free.
            return mean_.append(ratio);
        }

    private:
        detail::RollingMean mean_;
    };

} // namespace screamer

#endif // SCREAMER_AMIHUD_ILLIQUIDITY_H
