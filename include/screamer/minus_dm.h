#ifndef SCREAMER_MINUS_DM_H
#define SCREAMER_MINUS_DM_H

// MinusDM: Wilder's smoothed negative directional movement, -DM. The
// Wilder-smoothed run of downward moves (down = prev_low - low, counted when
// it exceeds the up move). Takes high and low. Shares DmiCore with ADX; -DM
// does not depend on close, so the node feeds close = high to satisfy
// DmiCore::update's NaN gate without affecting the +DM/-DM recurrence.

#include "screamer/common/functor_base.h"
#include "screamer/detail/dmi_core.h"

namespace screamer {

class MinusDM : public FunctorBase<MinusDM, 2, 1> {
public:
    explicit MinusDM(int window_size = 14) : core_(window_size) {}
    void reset() override { core_.reset(); }
    ResultTuple call(const InputArray& inputs) override {
        const double high = inputs[0];
        const double low = inputs[1];
        return core_.update(high, low, high).minus_dm;
    }
private:
    detail::DmiCore core_;
};

}  // namespace screamer

#endif  // SCREAMER_MINUS_DM_H
