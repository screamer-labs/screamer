#ifndef SCREAMER_AD_H
#define SCREAMER_AD_H

// AD: Accumulation / Distribution Line (Marc Chaikin). Cumulative
// volume weighted by where the close sits within the bar's range:
//
//     CLV[t] = ((close - low) - (high - close)) / (high - low)
//     AD[t]  = AD[t-1] + CLV * volume
//
// "Close location value" CLV is in [-1, +1]: +1 = close at high
// (full accumulation), -1 = close at low (full distribution).
//
// 4 -> 1 over (high, low, close, volume). Cumulative; no window.
// Returns 0 when high == low (flat bar; matches TA-Lib).

#include <cmath>
#include <limits>
#include "screamer/common/functor_base.h"

namespace screamer {

class AD : public FunctorBase<AD, 4, 1> {
public:
    AD() = default;

    void reset() override { ad_ = 0.0; }

    ResultTuple call(const InputArray& inputs) override {
        const double high   = inputs[0];
        const double low    = inputs[1];
        const double close  = inputs[2];
        const double volume = inputs[3];
        if (std::isnan(high) || std::isnan(low) || std::isnan(close) || std::isnan(volume)) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        const double range = high - low;
        if (range > 0.0) {
            const double clv = ((close - low) - (high - close)) / range;
            ad_ += clv * volume;
        }
        return ad_;
    }

private:
    double ad_ = 0.0;
};

}  // namespace screamer

#endif
