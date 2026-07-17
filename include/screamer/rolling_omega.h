#ifndef SCREAMER_ROLLING_OMEGA_H
#define SCREAMER_ROLLING_OMEGA_H

#include <limits>
#include <stdexcept>
#include "screamer/common/base.h"
#include "screamer/common/float_info.h"
#include "screamer/detail/rolling_sum.h"

namespace screamer {

    // RollingOmega: the Omega ratio (Keating-Shadwick 2002) over a trailing
    // window of returns about a threshold,
    //     omega = sum( (x - threshold)+ ) / sum( (threshold - x)+ ),
    // the total gain above the threshold divided by the total shortfall below it.
    // Omega uses the whole return distribution rather than just its mean and
    // variance, so it captures skew and fat tails a Sharpe ratio misses. A value
    // above 1 means gains outweigh losses at that threshold. A window with no
    // downside (zero denominator) has an undefined ratio and returns NaN. 1 -> 1;
    // composes two detail::RollingSum buffers. Like the other risk ratios
    // (RollingSharpe, RollingSortino) it uses a strict full window, so the output
    // is NaN until the window fills. nan_policy: ignore.
    class RollingOmega : public ScreamerBase {
    public:
        RollingOmega(int window_size, double threshold = 0.0)
            : window_size_(window_size),
              threshold_(threshold),
              sum_gain_(window_size, "strict"),
              sum_loss_(window_size, "strict")
        {
            if (window_size_ < 2) {
                throw std::invalid_argument("Window size must be 2 or more.");
            }
            reset();
        }

        void reset() override {
            sum_gain_.reset();
            sum_loss_.reset();
        }

        double process_scalar(double x) override {
            if (isnan2(x)) {
                return std::numeric_limits<double>::quiet_NaN();
            }
            const double excess = x - threshold_;
            const double gain = (excess > 0.0) ? excess : 0.0;
            const double loss = (excess < 0.0) ? -excess : 0.0;
            const double sum_gain = sum_gain_.append(gain);
            const double sum_loss = sum_loss_.append(loss);
            // During strict warmup the RollingSums return NaN, which propagates.
            if (isnan2(sum_gain) || isnan2(sum_loss)) {
                return std::numeric_limits<double>::quiet_NaN();
            }
            return (sum_loss > 0.0)
                ? sum_gain / sum_loss
                : std::numeric_limits<double>::quiet_NaN();   // no downside -> undefined
        }

    private:
        const int window_size_;
        const double threshold_;
        detail::RollingSum sum_gain_;
        detail::RollingSum sum_loss_;
    };

} // namespace screamer

#endif // SCREAMER_ROLLING_OMEGA_H
