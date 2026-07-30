#ifndef SCREAMER_MAMA_H
#define SCREAMER_MAMA_H

// MAMA: MESA Adaptive Moving Average (John Ehlers). The smoothing factor adapts
// to the rate of change of the instantaneous phase: when the phase moves fast
// (a new cycle), it tracks quickly; when the phase stalls, it smooths heavily.
// FAMA (following adaptive MA) is a second, slower pass. Returns (mama, fama).
// Reuses detail::HilbertCycle for the phase.

#include <cmath>
#include <limits>
#include "screamer/common/functor_base.h"
#include "screamer/detail/hilbert_cycle.h"

namespace screamer {

class MAMA : public FunctorBase<MAMA, 1, 2> {
public:
    explicit MAMA(double fast_limit = 0.5, double slow_limit = 0.05)
        : fast_(fast_limit), slow_(slow_limit) {}

    void reset() override {
        engine_.reset();
        prev_phase_ = std::numeric_limits<double>::quiet_NaN();
        mama_ = fama_ = std::numeric_limits<double>::quiet_NaN();
    }

    ResultTuple call(const InputArray& inputs) override {
        const double price = inputs[0];
        engine_.update(price);
        const double phase = engine_.phase();
        const double nan = std::numeric_limits<double>::quiet_NaN();
        if (std::isnan(phase)) {
            return std::make_tuple(nan, nan);
        }
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
