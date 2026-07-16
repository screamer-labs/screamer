#ifndef SCREAMER_HAWKES_INTENSITY_H
#define SCREAMER_HAWKES_INTENSITY_H

#include <limits>
#include <stdexcept>
#include "screamer/common/base.h"
#include "screamer/common/float_info.h"

namespace screamer {

    // Conditional intensity of an exponential-kernel Hawkes process: a
    // self-exciting model where each event raises the near-term rate of further
    // events (order-flow clustering / momentum).
    //
    //   lambda_t = mu + kappa_t,   kappa_{t+1} = decay * (kappa_t + alpha * x_t),
    //   kappa_0 = 0.
    //
    // Causal: lambda_t is emitted before x_t is folded into the state, so it
    // depends on x only up to t-1. A NaN event mark is ignored (emit NaN, leave
    // the state untouched), so it does not poison the recursion.
    class HawkesIntensity : public ScreamerBase {
    public:
        HawkesIntensity(double decay = 0.9, double alpha = 1.0, double mu = 0.0)
            : decay_(decay), alpha_(alpha), mu_(mu)
        {
            if (!(decay_ > 0.0 && decay_ < 1.0)) {
                throw std::invalid_argument("decay must be between 0 and 1 (exclusive)");
            }
            reset();
        }

        void reset() override {
            kappa_ = 0.0;
        }

        double process_scalar(double newValue) override {
            // NaN policy "ignore": emit NaN, leave the recursion state untouched.
            if (isnan2(newValue)) {
                return std::numeric_limits<double>::quiet_NaN();
            }
            double lambda = mu_ + kappa_;                        // emit before update (causal)
            kappa_ = decay_ * (kappa_ + alpha_ * newValue);
            return lambda;
        }

    private:
        const double decay_;
        const double alpha_;
        const double mu_;
        double kappa_ = 0.0;
    };

} // namespace screamer

#endif // SCREAMER_HAWKES_INTENSITY_H
