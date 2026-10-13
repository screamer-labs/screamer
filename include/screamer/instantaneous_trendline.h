#ifndef SCREAMER_INSTANTANEOUS_TRENDLINE_H
#define SCREAMER_INSTANTANEOUS_TRENDLINE_H

// InstantaneousTrendline: Ehlers' instantaneous trendline (Rocket Science for
// Traders, "The Instantaneous Trendline"; TA-Lib HT_TRENDLINE). The mean of
// the raw price over one measured dominant cycle, which removes that cycle,
// then a 4-tap WMA:
//
//   N         = int(SmoothPeriod + .5)
//   ITrend    = mean(price[t-N+1 .. t])
//   trendline = (4 ITrend + 3 ITrend[1] + 2 ITrend[2] + ITrend[3]) / 10
//
// On a ramp it lags by (N - 1) / 2 + 1 samples. See detail/ehlers_sinewave.h.

#include <cmath>
#include <limits>
#include "screamer/common/functor_base.h"
#include "screamer/detail/hilbert_cycle.h"
#include "screamer/detail/ehlers_sinewave.h"

namespace screamer {

class InstantaneousTrendline : public FunctorBase<InstantaneousTrendline, 1, 1> {
public:
    InstantaneousTrendline() = default;

    void reset() override {
        engine_.reset();
        trendline_.reset();
    }

    ResultTuple call(const InputArray& inputs) override {
        const double x = inputs[0];
        const double nan = std::numeric_limits<double>::quiet_NaN();
        if (std::isnan(x)) return nan;  // nan_policy ignore: state untouched.
        trendline_.prepare(x);
        engine_.update(x);
        trendline_.finish(engine_.raw_smooth_period());
        if (!engine_.valid()) return nan;
        return trendline_.value();
    }

private:
    detail::HilbertCycle engine_;
    detail::CycleTrendline trendline_;
};

}  // namespace screamer

#endif  // SCREAMER_INSTANTANEOUS_TRENDLINE_H
