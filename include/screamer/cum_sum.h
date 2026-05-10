#ifndef SCREAMER_CUM_SUM_H
#define SCREAMER_CUM_SUM_H

// CumSum: running sum from sample 0 to the current sample. O(1) memory.
// NaN propagates: once an input is NaN, all subsequent outputs are NaN.
// Matches numpy.cumsum behavior (which differs from pandas Series.cumsum
// where skipna=True is the default).

#include "screamer/common/base.h"

namespace screamer {

class CumSum : public ScreamerBase {
public:
    CumSum() { reset(); }

    void reset() override { sum_ = 0.0; }

    double process_scalar(double x) override {
        sum_ += x;     // NaN naturally propagates through addition
        return sum_;
    }

private:
    double sum_ = 0.0;
};

}  // namespace screamer

#endif  // SCREAMER_CUM_SUM_H
