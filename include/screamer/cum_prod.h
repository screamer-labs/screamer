#ifndef SCREAMER_CUM_PROD_H
#define SCREAMER_CUM_PROD_H

// CumProd: running product from sample 0 to the current sample. O(1) memory.
// Once 0 is multiplied in, the running product stays at 0.
//
// NaN policy "ignore" (see docs/nan_policy.md): a NaN input is skipped --
// output is NaN at that index, the running product is unchanged. This
// matches pandas Series.cumprod(skipna=True) and differs from numpy.cumprod.

#include <cmath>
#include <limits>
#include "screamer/common/base.h"

namespace screamer {

class CumProd : public ScreamerBase {
public:
    CumProd() { reset(); }

    void reset() override { prod_ = 1.0; }

    double process_scalar(double x) override {
        if (std::isnan(x)) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        prod_ *= x;
        return prod_;
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
        double prod = prod_;
        skip_nan_column(y, dy, x, dx, size, [&prod](double v) { prod *= v; return prod; });
        prod_ = prod;
    }

    double prod_ = 1.0;
};

}  // namespace screamer

#endif  // SCREAMER_CUM_PROD_H
