#ifndef SCREAMER_HILBERT_PHASOR_H
#define SCREAMER_HILBERT_PHASOR_H

// HilbertPhasor: the in-phase and quadrature components of the analytic signal
// (real and imaginary parts), from Ehlers' Hilbert transform. See
// detail/hilbert_cycle.h.

#include "screamer/common/functor_base.h"
#include "screamer/detail/hilbert_cycle.h"

namespace screamer {

class HilbertPhasor : public FunctorBase<HilbertPhasor, 1, 2> {
public:
    HilbertPhasor() = default;
    void reset() override { engine_.reset(); }
    ResultTuple call(const InputArray& inputs) override {
        engine_.update(inputs[0]);
        return std::make_tuple(engine_.inphase(), engine_.quadrature());
    }
private:
    detail::HilbertCycle engine_;
};

}  // namespace screamer

#endif  // SCREAMER_HILBERT_PHASOR_H
