#ifndef SCREAMER_CYCLE_SINE_H
#define SCREAMER_CYCLE_SINE_H

// CycleSine: the sinewave indicator, (sine, leadsine) = sin(phase) and
// sin(phase + 45 degrees), from the instantaneous phase of the analytic
// signal. See detail/hilbert_cycle.h.

#include <cmath>
#include <limits>
#include "screamer/common/functor_base.h"
#include "screamer/detail/hilbert_cycle.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace screamer {

class CycleSine : public FunctorBase<CycleSine, 1, 2> {
public:
    CycleSine() = default;
    void reset() override { engine_.reset(); }
    ResultTuple call(const InputArray& inputs) override {
        engine_.update(inputs[0]);
        const double ph = engine_.phase();
        if (std::isnan(ph)) {
            const double n = std::numeric_limits<double>::quiet_NaN();
            return std::make_tuple(n, n);
        }
        const double r = ph * M_PI / 180.0;
        return std::make_tuple(std::sin(r), std::sin(r + M_PI / 4.0));
    }
private:
    detail::HilbertCycle engine_;
};

}  // namespace screamer

#endif  // SCREAMER_CYCLE_SINE_H
