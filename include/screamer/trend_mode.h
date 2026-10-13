#ifndef SCREAMER_TREND_MODE_H
#define SCREAMER_TREND_MODE_H

// TrendMode: Ehlers' trend-versus-cycle mode (Rocket Science for Traders,
// "Identifying Market Modes"; TA-Lib HT_TRENDMODE). Outputs 1.0 (trend) or
// 0.0 (cycle). Trend is assumed, then:
//   - a crossing of sine and leadsine (CycleSine) restarts the trend count
//     and gives cycle; fewer than SmoothPeriod / 2 samples since the last
//     crossing also give cycle;
//   - a DCPhase advance between .67 and 1.5 times 360 / SmoothPeriod per
//     sample (the phase rotates at the cycle rate) gives cycle;
//   - the smoothed price at least 1.5% away from the instantaneous
//     trendline gives trend, overriding both.
// See detail/ehlers_sinewave.h.

#include <cmath>
#include <limits>
#include "screamer/common/functor_base.h"
#include "screamer/detail/hilbert_cycle.h"
#include "screamer/detail/ehlers_sinewave.h"

namespace screamer {

class TrendMode : public FunctorBase<TrendMode, 1, 1> {
public:
    TrendMode() = default;
    void reset() override {
        engine_.reset();
        dc_.reset();
        trendline_.reset();
        days_in_trend_ = 0;
    }
    ResultTuple call(const InputArray& inputs) override {
        const double x = inputs[0];
        const double nan = std::numeric_limits<double>::quiet_NaN();
        if (std::isnan(x)) return nan;  // nan_policy ignore: state untouched.
        engine_.update(x);
        const double sp = engine_.raw_smooth_period();
        dc_.update(engine_.smooth_price(), sp);
        dc_.advance_degrees();
        trendline_.update(x, sp);

        // TA-Lib's rule, as bitwise logic: the inputs to these tests vary
        // at random on noisy data, and mispredicted branches would stall
        // the engine. Trend is assumed, then:
        // a crossing, (sine > leadsine && prev sine <= prev leadsine) or
        // (sine < leadsine && prev sine >= prev leadsine), restarts the count;
        const int side = dc_.side(), prev = dc_.prev_side();
        const bool cross = ((side > 0) & (prev <= 0)) | ((side < 0) & (prev >= 0));
        days_in_trend_ = cross ? 1 : days_in_trend_ + 1;
        // cycle within half a period of the last crossing;
        const bool young = days_in_trend_ < 0.5 * sp;
        // cycle while the phase advances at .67 to 1.5 times the cycle rate;
        const double dphase = dc_.phase() - dc_.prev_phase();
        const double rate = dc_.rate();
        const bool rotating = (sp > 0.0) & (dphase > 0.67 * rate) & (dphase < 1.5 * rate);
        // trend whenever the smoothed price is 1.5% or more off the trendline,
        // |(smooth - trendline) / trendline| >= .015.
        const double tl = trendline_.value();
        const bool away = (tl != 0.0) & (std::abs(engine_.smooth_price() - tl) >= 0.015 * std::abs(tl));
        const bool trend = away | !(cross | young | rotating);

        if (!engine_.valid()) return nan;
        return trend ? 1.0 : 0.0;
    }

private:
    detail::HilbertCycle engine_;
    detail::DcPhase dc_;
    detail::CycleTrendline trendline_;
    long long days_in_trend_ = 0;
};

}  // namespace screamer

#endif  // SCREAMER_TREND_MODE_H
