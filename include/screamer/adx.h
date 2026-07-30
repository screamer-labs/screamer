#ifndef SCREAMER_ADX_H
#define SCREAMER_ADX_H

// ADX: Average Directional Index (Wilder, 1978). Measures trend
// strength (not direction). Returns (+DI, -DI, ADX) per step.
//
//     TR[t]   = max(H - L, |H - prev_C|, |L - prev_C|)
//     +DM[t]  = max(H - prev_H, 0) if H - prev_H > prev_L - L else 0
//     -DM[t]  = max(prev_L - L, 0) if prev_L - L > H - prev_H else 0
//
// TR / +DM / -DM are smoothed using TA-Lib's "ADX-flavour" Wilder:
// accumulate (timeperiod - 1) values during warmup, then apply the
// sum-form recurrence
//
//     prev = prev * (1 - 1/w) + new
//
// starting at index `window_size`. The +DI / -DI ratios cancel the
// 1/w scaling so the seed produces output identical to TA-Lib.
//
// ADX is Wilder-smoothed DX in the "average-form": SMA seed over the
// first `window_size` DX values (indices `window_size`..`2*window-1`),
// then standard ((w-1)*prev + new)/w recurrence.
//
// First valid +DI / -DI at sample index `window_size`. First valid
// ADX at sample index `2*window_size - 1`. Bit-exact to TA-Lib.
//
// The directional-movement math itself lives in detail::DmiCore, shared
// with the standalone PlusDI / MinusDI / PlusDM / MinusDM / DX operators.

#include <tuple>
#include "screamer/common/functor_base.h"
#include "screamer/detail/dmi_core.h"

namespace screamer {

class ADX : public FunctorBase<ADX, 3, 3> {
public:
    explicit ADX(int window_size = 14) : core_(window_size) {}

    void reset() override { core_.reset(); }

    ResultTuple call(const InputArray& inputs) override {
        const detail::DmiResult r = core_.update(inputs[0], inputs[1], inputs[2]);
        return std::make_tuple(r.plus_di, r.minus_di, r.adx);
    }

private:
    detail::DmiCore core_;
};

}  // namespace screamer

#endif
