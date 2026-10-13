#ifndef SCREAMER_HILBERT_PHASOR_H
#define SCREAMER_HILBERT_PHASOR_H

// HilbertPhasor: Ehlers' in-phase and quadrature components I1 and Q1 (the
// pair TA-Lib HT_PHASOR returns), each divided by its filter gain at the
// measured period so a unit sine reads 1 at periods from about 7 to 50. Q leads
// I by a quarter cycle. See detail/hilbert_cycle.h.

#include "screamer/common/functor_base.h"
#include "screamer/detail/hilbert_cycle.h"

namespace screamer {

class HilbertPhasor : public FunctorBase<HilbertPhasor, 1, 2> {
public:
    HilbertPhasor() = default;
    void reset() override { engine_.reset(); }
    ResultTuple call(const InputArray& inputs) override {
        engine_.update(inputs[0]);
        double i, q;
        engine_.phasor(i, q);
        return std::make_tuple(i, q);
    }
private:
    detail::HilbertCycle engine_;
};

}  // namespace screamer

#endif  // SCREAMER_HILBERT_PHASOR_H
