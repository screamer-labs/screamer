#ifndef SCREAMER_CYCLE_PHASE_H
#define SCREAMER_CYCLE_PHASE_H

// CyclePhase: instantaneous phase atan2(Q1, I1) in degrees, wrapped to
// [0, 360). It decreases by about 360 / period per sample, the direction of
// Ehlers' MAMA phase. See detail/hilbert_cycle.h.

#include "screamer/common/functor_base.h"
#include "screamer/detail/hilbert_cycle.h"

namespace screamer {

class CyclePhase : public FunctorBase<CyclePhase, 1, 1> {
public:
    CyclePhase() = default;
    void reset() override { engine_.reset(); }
    ResultTuple call(const InputArray& inputs) override {
        engine_.update(inputs[0]);
        return engine_.phase();
    }
private:
    detail::HilbertCycle engine_;
};

}  // namespace screamer

#endif  // SCREAMER_CYCLE_PHASE_H
