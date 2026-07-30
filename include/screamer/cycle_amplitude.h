#ifndef SCREAMER_CYCLE_AMPLITUDE_H
#define SCREAMER_CYCLE_AMPLITUDE_H

// CycleAmplitude: instantaneous amplitude (envelope) of the analytic signal,
// its magnitude sqrt(I^2 + Q^2). A general envelope detector. See
// detail/hilbert_cycle.h.

#include "screamer/common/functor_base.h"
#include "screamer/detail/hilbert_cycle.h"

namespace screamer {

class CycleAmplitude : public FunctorBase<CycleAmplitude, 1, 1> {
public:
    CycleAmplitude() = default;
    void reset() override { engine_.reset(); }
    ResultTuple call(const InputArray& inputs) override {
        engine_.update(inputs[0]);
        return engine_.amplitude();
    }
private:
    detail::HilbertCycle engine_;
};

}  // namespace screamer

#endif  // SCREAMER_CYCLE_AMPLITUDE_H
