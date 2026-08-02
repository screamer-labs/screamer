#ifndef SCREAMER_ROLLING_MIN_H
#define SCREAMER_ROLLING_MIN_H

#include <cmath>
#include <limits>
#include "screamer/common/base.h"
#include "screamer/detail/block_extremum.h"
#include "screamer/detail/monotonic_deque.h"
#include <cstddef>

namespace screamer {

class RollingMin : public ScreamerBase {
public:
    explicit RollingMin(int window_size) : deque_(window_size) {}

    void reset() override { deque_.reset(); }

    // See RollingMax: the block decomposition computes the same values with no
    // data-dependent branching, and falls back when it cannot apply.
    void process_array_no_stride(double* y, const double* x, size_t size) override {
        if (deque_.samples_seen() != 0 || detail::has_nan(x, size)) {
            ScreamerBase::process_array_no_stride(y, x, size);
            return;
        }
        detail::block_extremum<false>(y, x, size, deque_.window_size());
        const size_t window = static_cast<size_t>(deque_.window_size());
        for (size_t i = (size > window) ? size - window : 0; i < size; ++i) {
            deque_.append(x[i]);
        }
    }

private:
    double process_scalar(double newValue) override {
        // NaN policy "ignore": leave the deque state untouched and emit NaN.
        if (std::isnan(newValue)) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        return deque_.append(newValue);
    }

    detail::MinDeque deque_;
};

}  // namespace screamer

#endif
