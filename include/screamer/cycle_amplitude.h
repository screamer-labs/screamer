#ifndef SCREAMER_CYCLE_AMPLITUDE_H
#define SCREAMER_CYCLE_AMPLITUDE_H

// CycleAmplitude: instantaneous amplitude (envelope) of the dominant cycle,
// the magnitude of the HilbertPhasor pair. Each component is divided by its
// filter gain at the measured period, so a sine of a period from about 7 to
// 50 reads its own amplitude (Ehlers' |(I1, Q1)| at exact unit gain). See detail/hilbert_cycle.h.

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
