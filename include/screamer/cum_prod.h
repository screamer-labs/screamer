#ifndef SCREAMER_CUM_PROD_H
#define SCREAMER_CUM_PROD_H

// CumProd: running product from sample 0 to the current sample. O(1) memory.
// NaN propagates: once an input is NaN, all subsequent outputs are NaN.
// Once 0 is multiplied in, the running product stays at 0.
// Matches numpy.cumprod behavior.

#include "screamer/common/base.h"

namespace screamer {

class CumProd : public ScreamerBase {
public:
    CumProd() { reset(); }

    void reset() override { prod_ = 1.0; }

    double process_scalar(double x) override {
        prod_ *= x;    // NaN naturally propagates through multiplication
        return prod_;
    }

private:
    double prod_ = 1.0;
};

}  // namespace screamer

#endif  // SCREAMER_CUM_PROD_H
