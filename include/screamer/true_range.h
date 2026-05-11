#ifndef SCREAMER_TRUE_RANGE_H
#define SCREAMER_TRUE_RANGE_H

// TrueRange (Wilder, 1978). Per-bar OHLC-aware range:
//
//     TR[t] = max(high - low,
//                 |high - close[t-1]|,
//                 |low  - close[t-1]|)
//
// 3 -> 1 over (high, low, close). The previous close captures
// overnight gaps; if it is missing (t = 0) TA-Lib returns NaN, and
// we follow.

#include <algorithm>
#include <cmath>
#include <limits>
#include "screamer/common/float_info.h"
#include "screamer/common/functor_base.h"

namespace screamer {

class TrueRange : public FunctorBase<TrueRange, 3, 1> {
public:
    TrueRange() = default;

    void reset() override {
        prev_close_ = std::numeric_limits<double>::quiet_NaN();
    }

    ResultTuple call(const InputArray& inputs) override {
        const double high  = inputs[0];
        const double low   = inputs[1];
        const double close = inputs[2];
        if (isnan2(prev_close_)) {
            prev_close_ = close;
            return std::numeric_limits<double>::quiet_NaN();
        }
        const double r = std::max({
            high - low,
            std::abs(high - prev_close_),
            std::abs(low  - prev_close_),
        });
        prev_close_ = close;
        return r;
    }

private:
    double prev_close_ = std::numeric_limits<double>::quiet_NaN();
};

}  // namespace screamer

#endif
