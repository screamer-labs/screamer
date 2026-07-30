#ifndef SCREAMER_TREND_MODE_H
#define SCREAMER_TREND_MODE_H

// TrendMode: a trend-vs-cycle classifier. Outputs 1.0 when the dominant-cycle
// phase advance per sample is a small fraction of a full cycle for a sustained
// run (the cycle has stalled, so the series is trending), 0.0 when the phase
// rotates at the cycle rate. See detail/hilbert_cycle.h.

#include <cmath>
#include <limits>
#include "screamer/common/functor_base.h"
#include "screamer/detail/hilbert_cycle.h"

namespace screamer {

class TrendMode : public FunctorBase<TrendMode, 1, 1> {
public:
    explicit TrendMode(double phase_rate_frac = 0.5) : frac_(phase_rate_frac) {}
    void reset() override {
        engine_.reset();
        prev_phase_ = std::numeric_limits<double>::quiet_NaN();
    }
    ResultTuple call(const InputArray& inputs) override {
        engine_.update(inputs[0]);
        const double ph = engine_.phase();
        const double per = engine_.period();
        const double nan = std::numeric_limits<double>::quiet_NaN();
        if (std::isnan(ph) || std::isnan(per) || per <= 0.0) {
            // nan_policy "ignore": leave prev_phase_ untouched. During
            // warm-up it is already NaN; on a mid-stream NaN input it must
            // keep the last valid phase so the next finite sample computes
            // the correct delta, as if this sample had not occurred.
            return nan;
        }
        double out = 0.0;
        if (!std::isnan(prev_phase_)) {
            double dphase = ph - prev_phase_;
            // Unwrap to [-180, 180].
            while (dphase > 180.0) dphase -= 360.0;
            while (dphase < -180.0) dphase += 360.0;
            const double expected = 360.0 / per;  // per-sample advance if cycling.
            out = (std::abs(dphase) < frac_ * expected) ? 1.0 : 0.0;
        }
        prev_phase_ = ph;
        return out;
    }

private:
    detail::HilbertCycle engine_;
    double frac_ = 0.5;
    double prev_phase_ = std::numeric_limits<double>::quiet_NaN();
};

}  // namespace screamer

#endif  // SCREAMER_TREND_MODE_H
