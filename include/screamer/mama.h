#ifndef SCREAMER_MAMA_H
#define SCREAMER_MAMA_H

// MAMA: MESA Adaptive Moving Average (Ehlers, "MESA Adaptive Moving Averages",
// S&C 2001, and Rocket Science for Traders; TA-Lib MAMA). The phase is
// Ehlers' arctan(Q1 / I1) in degrees, in (-90, 90), so it wraps every half
// cycle. DeltaPhase = Phase[1] - Phase, floored at 1, and
// alpha = clip(fast_limit / DeltaPhase, slow_limit, fast_limit): a phase
// that rotates at the cycle rate gives a small alpha (heavy smoothing), and
// alpha reaches fast_limit when the phase stalls or wraps. FAMA (following
// adaptive MA) is a second pass at alpha / 2. Returns (mama, fama). Reuses
// detail::HilbertCycle for I1 and Q1.

#include "screamer/detail/test_access.h"
#include "screamer/common/finite_settings.h"
#include <cmath>
#include <limits>
#include "screamer/common/functor_base.h"
#include "screamer/detail/hilbert_cycle.h"

namespace screamer {

class MAMA : public FunctorBase<MAMA, 1, 2> {
    friend struct detail::TestAccess;   // tests read the event counters (test_access.h)
public:
    explicit MAMA(double fast_limit = 0.5, double slow_limit = 0.05)
        : fast_(fast_limit), slow_(slow_limit) {
        require_finite("fast_limit", fast_limit);
        require_finite("slow_limit", slow_limit);
    }

    void reset() override {
        engine_.reset();
        prev_phase_ = std::numeric_limits<double>::quiet_NaN();
        mama_ = fama_ = std::numeric_limits<double>::quiet_NaN();
    }

    ResultTuple call(const InputArray& inputs) override {
        const double price = inputs[0];
        engine_.update(price);
        const double nan = std::numeric_limits<double>::quiet_NaN();
        if (!engine_.valid()) {
            return std::make_tuple(nan, nan);
        }
        const double i1 = engine_.raw_i1();
        const double phase = (i1 != 0.0) ? std::atan(engine_.raw_q1() / i1) * 180.0 / M_PI : 0.0;
        // DeltaPhase: how far the phase advanced since the last sample. Ehlers
        // measures it as prev_phase - phase and floors it at 1 degree.
        double dphase = std::isnan(prev_phase_) ? 1.0 : (prev_phase_ - phase);
        prev_phase_ = phase;
        if (dphase < 1.0) dphase = 1.0;
        double alpha = fast_ / dphase;
        if (alpha < slow_) alpha = slow_;
        if (alpha > fast_) alpha = fast_;
        if (std::isnan(mama_)) {
            mama_ = price;
            fama_ = price;
        } else {
            mama_ = alpha * price + (1.0 - alpha) * mama_;
            fama_ = 0.5 * alpha * mama_ + (1.0 - 0.5 * alpha) * fama_;
        }
        return std::make_tuple(mama_, fama_);
    }

private:
    detail::HilbertCycle engine_;
    const double fast_;
    const double slow_;
    double prev_phase_ = std::numeric_limits<double>::quiet_NaN();
    double mama_ = std::numeric_limits<double>::quiet_NaN();
    double fama_ = std::numeric_limits<double>::quiet_NaN();
};

}  // namespace screamer

#endif  // SCREAMER_MAMA_H
