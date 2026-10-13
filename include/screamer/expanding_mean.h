#ifndef SCREAMER_EXPANDING_MEAN_H
#define SCREAMER_EXPANDING_MEAN_H

// ExpandingMean: running mean over the whole history since the last reset().
// O(1) memory, no window, no start_policy. Mirrors RollingMean but keeps an
// unbounded running sum instead of a windowed detail::RollingSum.
//
// NaN policy "ignore" (see docs/nan_policy.md): a NaN input is skipped --
// output is NaN at that index and the running sum / count are unchanged.
// Matches pandas Series.expanding().mean().

#include <cmath>
#include <cstdint>
#include <limits>
#include "screamer/common/base.h"

namespace screamer {

class ExpandingMean : public ScreamerBase {
public:
    ExpandingMean() { reset(); }

    void reset() override {
        sum_x_ = 0.0;
        n_ = 0;
    }

    double process_scalar(double x) override {
        if (std::isnan(x)) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        sum_x_ += x;
        ++n_;
        return sum_x_ / n_;
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
        double sum_x = sum_x_;
        std::int64_t n = n_;
        skip_nan_column(y, dy, x, dx, size, [&sum_x, &n](double v) {
            sum_x += v;
            ++n;
            return sum_x / n;
        });
        sum_x_ = sum_x;
        n_ = n;
    }

    double sum_x_ = 0.0;
    std::int64_t n_ = 0;
};

}  // namespace screamer

#endif  // SCREAMER_EXPANDING_MEAN_H
