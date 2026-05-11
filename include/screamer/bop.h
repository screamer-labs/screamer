#ifndef SCREAMER_BOP_H
#define SCREAMER_BOP_H

// BOP: Balance of Power (Igor Livshin). 4 -> 1 functor on
// (open, high, low, close):
//
//     BOP[t] = (close - open) / (high - low)
//
// Measures whether the bar closes near its high or its low; values
// are in [-1, 1]. Stateless, no warmup. Returns 0 when high == low.

#include "screamer/common/functor_base.h"

namespace screamer {

class BOP : public FunctorBase<BOP, 4, 1> {
public:
    BOP() = default;

    ResultTuple call(const InputArray& inputs) override {
        const double open  = inputs[0];
        const double high  = inputs[1];
        const double low   = inputs[2];
        const double close = inputs[3];
        const double range = high - low;
        if (range <= 0.0) {
            return 0.0;
        }
        return (close - open) / range;
    }
};

}  // namespace screamer

#endif
