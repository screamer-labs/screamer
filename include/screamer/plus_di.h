#ifndef SCREAMER_PLUS_DI_H
#define SCREAMER_PLUS_DI_H

// PlusDI: Wilder's +DI (positive directional indicator), 100 times the
// smoothed positive directional movement divided by the smoothed true range.
// Measures the strength of upward movement. Shares DmiCore with ADX.

#include "screamer/common/functor_base.h"
#include "screamer/detail/dmi_core.h"

namespace screamer {

class PlusDI : public FunctorBase<PlusDI, 3, 1> {
public:
    explicit PlusDI(int window_size = 14) : core_(window_size) {}
    void reset() override { core_.reset(); }
    ResultTuple call(const InputArray& inputs) override {
        return core_.update(inputs[0], inputs[1], inputs[2]).plus_di;
    }
private:
    detail::DmiCore core_;
};

}  // namespace screamer

#endif  // SCREAMER_PLUS_DI_H
