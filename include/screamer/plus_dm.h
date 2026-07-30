#ifndef SCREAMER_PLUS_DM_H
#define SCREAMER_PLUS_DM_H

// PlusDM: Wilder's smoothed positive directional movement, +DM. The
// Wilder-smoothed run of upward moves (up = high - prev_high, counted when it
// exceeds the down move). Takes high and low. Shares DmiCore with ADX; +DM
// does not depend on close, so the node feeds close = high to satisfy
// DmiCore::update's NaN gate without affecting the +DM/-DM recurrence.

#include "screamer/common/functor_base.h"
#include "screamer/detail/dmi_core.h"

namespace screamer {

class PlusDM : public FunctorBase<PlusDM, 2, 1> {
public:
    explicit PlusDM(int window_size = 14) : core_(window_size) {}
    void reset() override { core_.reset(); }
    ResultTuple call(const InputArray& inputs) override {
        const double high = inputs[0];
        const double low = inputs[1];
        return core_.update(high, low, high).plus_dm;
    }
private:
    detail::DmiCore core_;
};

}  // namespace screamer

#endif  // SCREAMER_PLUS_DM_H
