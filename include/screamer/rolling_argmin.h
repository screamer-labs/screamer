#ifndef SCREAMER_ROLLING_ARGMIN_H
#define SCREAMER_ROLLING_ARGMIN_H

// RollingArgmin: position of the rolling minimum within the current
// window. Convention: 0 = oldest sample in the window, window_size-1 =
// newest. Matches numpy.argmin applied to the window slice and pandas'
// .rolling(w).apply(np.argmin).
//
// Same monotonic-deque algorithm as RollingMin -- amortised O(1) per
// step. We just expose the front element's window offset instead of
// its value.

#include <cmath>
#include <limits>
#include "screamer/common/base.h"
#include "screamer/detail/block_extremum.h"
#include "screamer/detail/monotonic_deque.h"
#include <cstddef>

namespace screamer {

class RollingArgmin : public ScreamerBase {
public:
    explicit RollingArgmin(int window_size) : deque_(window_size) {}

    void reset() override { deque_.reset(); }

    // See RollingMax: the block decomposition, carrying indices. Falls back
    // when the operator holds state or the input contains NaN.
    void process_array_no_stride(double* y, const double* x, size_t size) override {
        if (deque_.samples_seen() != 0 || detail::has_nan(x, size)) {
            ScreamerBase::process_array_no_stride(y, x, size);
            return;
        }
        detail::block_arg_extremum<false>(y, x, size, deque_.window_size());
        const size_t window = static_cast<size_t>(deque_.window_size());
        for (size_t i = (size > window) ? size - window : 0; i < size; ++i) {
            deque_.append(x[i]);
        }
    }

private:
    double process_scalar(double newValue) override {
        if (std::isnan(newValue)) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        deque_.append(newValue);
        return static_cast<double>(deque_.front_window_offset());
    }

    detail::MinDeque deque_;
};

}  // namespace screamer

#endif
