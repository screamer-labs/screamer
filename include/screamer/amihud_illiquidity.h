#ifndef SCREAMER_AMIHUD_ILLIQUIDITY_H
#define SCREAMER_AMIHUD_ILLIQUIDITY_H

#include <cmath>
#include <cstddef>
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
    class AmihudIlliquidity : public StepperFunctorBase<AmihudIlliquidity, 2, 1> {
    public:
        AmihudIlliquidity(int window_size, const std::string& start_policy = "strict")
            : mean_(window_size, start_policy)
        {
        }

        void reset() override {
            mean_.reset();
        }

        // One event on the window's state, for both the batch column loop and
        // call() (StepperFunctorBase): the running values stay in registers
        // in a batch call. The step is detail::rolling_mean_step on the ratio,
        // which ignores a NaN ratio (state untouched, NaN out), giving the
        // "ignore" behavior.
        struct Stepper {
            double* buf;
            std::size_t cap;
            std::size_t index;
            std::size_t count;
            double sum;
            detail::StartPolicy policy;

            inline ResultTuple call(const InputArray& inputs) {
                const double ret = inputs[0];
                const double notional = inputs[1];
                if (isnan2(ret) || isnan2(notional) || notional == 0.0) {
                    return std::numeric_limits<double>::quiet_NaN();
                }
                const double ratio = std::abs(ret) / notional;
                return detail::rolling_mean_step(buf, cap, index, count, sum, policy, ratio);
            }
        };
        Stepper stepper() {
            return Stepper{mean_.buffer_data(), mean_.capacity(), mean_.index(),
                           mean_.count(), mean_.sum(), mean_.start_policy()};
        }
        void store(const Stepper& k) { mean_.restore(k.index, k.count, k.sum); }

    private:
        detail::RollingMean mean_;
    };

} // namespace screamer

#endif // SCREAMER_AMIHUD_ILLIQUIDITY_H
