#ifndef SCREAMER_OBV_H
#define SCREAMER_OBV_H

// OBV: On-Balance Volume (Joseph Granville, 1963). Cumulative signed
// volume:
//
//     OBV[t] = OBV[t-1] + sign(close - prev_close) * volume[t]
//
// 2 -> 1 over (close, volume). Cumulative since reset; no window.
// At t=0 the OBV is conventionally 0 (TA-Lib starts at 0). Matches
// TA-Lib's OBV.

#include <limits>
#include "screamer/common/float_info.h"
#include "screamer/common/functor_base.h"

namespace screamer {

class OBV : public FunctorBase<OBV, 2, 1> {
public:
    OBV() = default;

    void reset() override {
        obv_ = 0.0;
        prev_close_ = std::numeric_limits<double>::quiet_NaN();
    }

    ResultTuple call(const InputArray& inputs) override {
        const double close  = inputs[0];
        const double volume = inputs[1];
        if (isnan2(close) || isnan2(volume)) {
            // NaN policy "ignore": leave running OBV and prev_close alone.
            return std::numeric_limits<double>::quiet_NaN();
        }
        if (isnan2(prev_close_)) {
            // Seed: TA-Lib starts OBV at volume[0].
            obv_ = volume;
        } else if (close > prev_close_) {
            obv_ += volume;
        } else if (close < prev_close_) {
            obv_ -= volume;
        }
        // close == prev_close: OBV unchanged.
        prev_close_ = close;
        return obv_;
    }

private:
    double obv_ = 0.0;
    double prev_close_ = std::numeric_limits<double>::quiet_NaN();
};

}  // namespace screamer

#endif
