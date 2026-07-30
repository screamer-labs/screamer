#ifndef SCREAMER_CYCLE_FREQUENCY_H
#define SCREAMER_CYCLE_FREQUENCY_H

// CycleFrequency: instantaneous frequency (cycles per sample), the reciprocal
// of the dominant cycle period. See detail/hilbert_cycle.h.

#include <cmath>
#include <limits>
#include "screamer/common/functor_base.h"
#include "screamer/detail/hilbert_cycle.h"

namespace screamer {

class CycleFrequency : public FunctorBase<CycleFrequency, 1, 1> {
public:
    CycleFrequency() = default;
    void reset() override { engine_.reset(); }
    ResultTuple call(const InputArray& inputs) override {
        engine_.update(inputs[0]);
        const double p = engine_.period();
        if (std::isnan(p) || p <= 0.0) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        return 1.0 / p;
    }
private:
    detail::HilbertCycle engine_;
};

}  // namespace screamer

#endif  // SCREAMER_CYCLE_FREQUENCY_H
