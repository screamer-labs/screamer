#ifndef SCREAMER_MAX_DRAWDOWN_H
#define SCREAMER_MAX_DRAWDOWN_H

// MaxDrawdown: the largest drawdown experienced since reset.
//
//     MaxDrawdown[t] = min over k <= t of Drawdown[k]
//                    = min over k <= t of (price[k] / CumMax(price)[k] - 1)
//
// Output is always in (-1, 0]. The "worst peak-to-trough loss so far".
//
// 1 -> 1. Composes Drawdown + CumMin. Cumulative; no window. For
// the windowed variant see RollingMaxDrawdown.

#include "screamer/common/base.h"
#include "screamer/cum_min.h"
#include "screamer/drawdown.h"

namespace screamer {

class MaxDrawdown : public ScreamerBase {
public:
    MaxDrawdown() = default;

    void reset() override {
        drawdown_.reset();
        cum_min_.reset();
    }

    double process_scalar(double price) override {
        const double dd = drawdown_.process_scalar(price);
        return cum_min_.process_scalar(dd);
    }

private:
    Drawdown drawdown_;
    CumMin cum_min_;
};

}  // namespace screamer

#endif
