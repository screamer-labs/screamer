#ifndef SCREAMER_INSTANTANEOUS_TRENDLINE_H
#define SCREAMER_INSTANTANEOUS_TRENDLINE_H

// InstantaneousTrendline: Ehlers' adaptive trendline. A 2-pole recursion whose
// smoothing factor alpha is set from the measured dominant cycle period, so the
// trendline follows the trend and removes the dominant cycle. Reuses
// detail::HilbertCycle for the period.
//
//   alpha = 2 / (period + 1)
//   it[t] = (alpha - alpha^2/4) price[t] + 0.5 alpha^2 price[t-1]
//           - (alpha - 0.75 alpha^2) price[t-2]
//           + 2 (1-alpha) it[t-1] - (1-alpha)^2 it[t-2]

#include <cmath>
#include <limits>
#include "screamer/common/functor_base.h"
#include "screamer/detail/hilbert_cycle.h"

namespace screamer {

class InstantaneousTrendline : public FunctorBase<InstantaneousTrendline, 1, 1> {
public:
    InstantaneousTrendline() = default;

    void reset() override {
        engine_.reset();
        p1_ = p2_ = std::numeric_limits<double>::quiet_NaN();
        it1_ = it2_ = std::numeric_limits<double>::quiet_NaN();
    }

    ResultTuple call(const InputArray& inputs) override {
        const double price = inputs[0];
        engine_.update(price);
        const double period = engine_.period();
        const double nan = std::numeric_limits<double>::quiet_NaN();
        if (std::isnan(period) || period <= 0.0) {
            // nan_policy "ignore": leave p1_/p2_/it1_/it2_ untouched so the
            // next finite sample resumes the recursion as if this sample had
            // not occurred.
            return nan;
        }
        const double a = 2.0 / (period + 1.0);
        double it;
        if (std::isnan(it1_) || std::isnan(p1_) || std::isnan(p2_)) {
            it = price;  // seed the recursion once the period is available.
        } else {
            it = (a - a * a / 4.0) * price + 0.5 * a * a * p1_
                 - (a - 0.75 * a * a) * p2_
                 + 2.0 * (1.0 - a) * it1_ - (1.0 - a) * (1.0 - a) * it2_;
        }
        p2_ = p1_;
        p1_ = price;
        it2_ = it1_;
        it1_ = it;
        return it;
    }

private:
    detail::HilbertCycle engine_;
    double p1_ = std::numeric_limits<double>::quiet_NaN();
    double p2_ = std::numeric_limits<double>::quiet_NaN();
    double it1_ = std::numeric_limits<double>::quiet_NaN();
    double it2_ = std::numeric_limits<double>::quiet_NaN();
};

}  // namespace screamer

#endif  // SCREAMER_INSTANTANEOUS_TRENDLINE_H
