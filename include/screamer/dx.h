#ifndef SCREAMER_DX_H
#define SCREAMER_DX_H

// DX: Wilder's directional index, 100 times the absolute difference of +DI and
// -DI over their sum. The pre-average input to ADX. Shares DmiCore with ADX.

#include "screamer/common/functor_base.h"
#include "screamer/detail/dmi_core.h"

namespace screamer {

class DX : public FunctorBase<DX, 3, 1> {
public:
    explicit DX(int window_size = 14) : core_(window_size) {}
    void reset() override { core_.reset(); }
    ResultTuple call(const InputArray& inputs) override {
        return core_.update(inputs[0], inputs[1], inputs[2]).dx;
    }
private:
    detail::DmiCore core_;
};

}  // namespace screamer

#endif  // SCREAMER_DX_H
