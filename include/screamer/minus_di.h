#ifndef SCREAMER_MINUS_DI_H
#define SCREAMER_MINUS_DI_H

// MinusDI: Wilder's -DI (negative directional indicator), 100 times the
// smoothed negative directional movement divided by the smoothed true range.
// Measures the strength of downward movement. Shares DmiCore with ADX.

#include "screamer/common/functor_base.h"
#include "screamer/detail/dmi_core.h"

namespace screamer {

class MinusDI : public FunctorBase<MinusDI, 3, 1> {
public:
    explicit MinusDI(int window_size = 14) : core_(window_size) {}
    void reset() override { core_.reset(); }
    ResultTuple call(const InputArray& inputs) override {
        return core_.update(inputs[0], inputs[1], inputs[2]).minus_di;
    }
private:
    detail::DmiCore core_;
};

}  // namespace screamer

#endif  // SCREAMER_MINUS_DI_H
