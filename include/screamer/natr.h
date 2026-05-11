#ifndef SCREAMER_NATR_H
#define SCREAMER_NATR_H

// NATR: Normalised Average True Range. ATR divided by close,
// expressed as a percentage:
//
//     NATR[t] = 100 * ATR[t] / close[t]
//
// 3 -> 1 over (high, low, close). Useful when comparing volatility
// across instruments with different price levels. Composes ATR.

#include <limits>
#include "screamer/atr.h"
#include "screamer/common/float_info.h"
#include "screamer/common/functor_base.h"

namespace screamer {

class NATR : public FunctorBase<NATR, 3, 1> {
public:
    explicit NATR(int window_size = 14) : atr_(window_size) {}

    void reset() override { atr_.reset(); }

    ResultTuple call(const InputArray& inputs) override {
        const double close = inputs[2];
        const double v = atr_.call(inputs);
        if (isnan2(v) || close == 0.0) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        return 100.0 * v / close;
    }

private:
    ATR atr_;
};

}  // namespace screamer

#endif
