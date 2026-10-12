#ifndef SCREAMER_CUM_SUM_H
#define SCREAMER_CUM_SUM_H

// CumSum: running sum from sample 0 to the current sample. O(1) memory.
//
// NaN policy "ignore" (see docs/nan_policy.md): a NaN input is skipped --
// output is NaN at that index, the running sum is unchanged. This matches
// pandas Series.cumsum(skipna=True) (the default) and differs from
// numpy.cumsum.

#include <cmath>
#include <limits>
#include "screamer/common/base.h"

namespace screamer {

class CumSum : public ScreamerBase {
public:
    CumSum() { reset(); }

    void reset() override { sum_ = 0.0; }

    double process_scalar(double x) override {
        if (std::isnan(x)) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        sum_ += x;
        return sum_;
    }

    void process_array_no_stride(double* y, const double* x, size_t size) override {
        run_column(y, 1, x, 1, size);
    }

    void process_array_stride(double* y, size_t dyi, const double* x, size_t dxi,
                              size_t size) override {
        run_column(y, static_cast<std::ptrdiff_t>(dyi), x, static_cast<std::ptrdiff_t>(dxi), size);
    }

private:
    // The batch loop, with the running state in a local (skip_nan_column).
    void run_column(double* y, std::ptrdiff_t dy, const double* x, std::ptrdiff_t dx,
                    size_t size) {
        double sum = sum_;
        skip_nan_column(y, dy, x, dx, size, [&sum](double v) { sum += v; return sum; });
        sum_ = sum;
    }

    double sum_ = 0.0;
};

}  // namespace screamer

#endif  // SCREAMER_CUM_SUM_H
