#ifndef SCREAMER_DOMINANT_CYCLE_H
#define SCREAMER_DOMINANT_CYCLE_H

// DominantCycle: the dominant cycle period (in samples) of a series, measured
// by Ehlers' homodyne discriminator. See detail/hilbert_cycle.h.

#include "screamer/common/functor_base.h"
#include "screamer/detail/hilbert_cycle.h"

namespace screamer {

class DominantCycle : public FunctorBase<DominantCycle, 1, 1> {
public:
    DominantCycle() = default;
    void reset() override { engine_.reset(); }
    ResultTuple call(const InputArray& inputs) override {
        engine_.update(inputs[0]);
        return engine_.period();
    }
private:
    detail::HilbertCycle engine_;
};

}  // namespace screamer

#endif  // SCREAMER_DOMINANT_CYCLE_H
