#ifndef SCREAMER_COMMON_SKIP_NAN_COLUMN_H
#define SCREAMER_COMMON_SKIP_NAN_COLUMN_H

#include <cstddef>
#include <limits>

namespace screamer {

// The batch loop of a one-input operator whose state the caller keeps in local
// variables: y = step(x) for each non-NaN x, and NaN (state untouched) for a
// NaN x. Strides are in elements. A NaN leaves the inner loop instead of
// selecting between the old and the new state, which would add the select to
// the latency of a recurrence. `step` must compute what process_scalar does,
// with the same expressions, so batch and stream stay identical.
template <class Step>
inline void skip_nan_column(double* y, std::ptrdiff_t dy, const double* x, std::ptrdiff_t dx,
                            std::size_t size, Step&& step) {
    std::size_t m = 0;
    while (true) {
        for (; m < size; ++m, x += dx, y += dy) {
            const double v = *x;
            if (v != v) break;
            *y = step(v);
        }
        if (m == size) return;
        *y = std::numeric_limits<double>::quiet_NaN();
        ++m;
        x += dx;
        y += dy;
    }
}

}  // namespace screamer
#endif
