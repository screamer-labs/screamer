#ifndef SCREAMER_CYCLE_SINE_H
#define SCREAMER_CYCLE_SINE_H

// CycleSine: Ehlers' sinewave indicator (Rocket Science for Traders, "The
// Sinewave Indicator"; TA-Lib HT_SINE). sine = sin(DCPhase) and leadsine =
// sin(DCPhase + 45 degrees), where DCPhase is the dominant cycle phase from a
// one-cycle DFT of the smoothed price. DCPhase increases with time, so
// leadsine leads sine by 1/8 cycle. See detail/ehlers_sinewave.h.

#include <cmath>
#include <limits>
#include "screamer/common/functor_base.h"
#include "screamer/detail/hilbert_cycle.h"
#include "screamer/detail/ehlers_sinewave.h"

namespace screamer {

class CycleSine : public FunctorBase<CycleSine, 1, 2> {
public:
    CycleSine() = default;
    void reset() override {
        engine_.reset();
        dc_.reset();
    }
    ResultTuple call(const InputArray& inputs) override {
        const double x = inputs[0];
        const double nan = std::numeric_limits<double>::quiet_NaN();
        if (std::isnan(x)) return std::make_tuple(nan, nan);  // nan_policy ignore.
        engine_.update(x);
        dc_.update(engine_.smooth_price(), engine_.raw_smooth_period());
        dc_.advance_sine();
        if (!engine_.valid()) return std::make_tuple(nan, nan);
        return std::make_tuple(dc_.sine(), dc_.leadsine());
    }
private:
    detail::HilbertCycle engine_;
    detail::DcPhase dc_;
};

}  // namespace screamer

#endif  // SCREAMER_CYCLE_SINE_H
