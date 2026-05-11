#ifndef SCREAMER_DRAWDOWN_H
#define SCREAMER_DRAWDOWN_H

// Drawdown: running drawdown from the cumulative peak since reset.
//
//     Drawdown[t] = price[t] / CumMax(price)[t] - 1     in (-1, 0]
//
// A flat or rising series has Drawdown[t] = 0. A 30% loss from the
// prior peak gives Drawdown[t] = -0.30. The maximum sustained loss
// over the entire history is min(Drawdown), which is what MaxDrawdown
// tracks.
//
// 1 -> 1. Composes screamer::CumMax. No warmup.
//
// MaxDrawdown is then CumMin(Drawdown), exposed as a separate class.

#include <cmath>
#include <limits>
#include "screamer/common/base.h"
#include "screamer/cum_max.h"

namespace screamer {

class Drawdown : public ScreamerBase {
public:
    Drawdown() { reset(); }

    void reset() override {
        cum_max_.reset();
    }

    double process_scalar(double price) override {
        const double peak = cum_max_.process_scalar(price);
        if (peak == 0.0) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        return price / peak - 1.0;
    }

private:
    CumMax cum_max_;
};

}  // namespace screamer

#endif
