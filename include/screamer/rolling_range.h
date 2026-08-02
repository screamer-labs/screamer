#ifndef SCREAMER_ROLLING_RANGE_H
#define SCREAMER_ROLLING_RANGE_H

// RollingRange: rolling max - min of a single stream.
//
// Algorithmically identical to RollingMinMax (two monotonic deques)
// followed by a subtract; we hold the two deques directly and return
// the difference, saving the tuple-allocation that the 1->2 dispatcher
// of RollingMinMax does per step. Memory and compute order are the
// same as RollingMinMax: amortised O(1) per step.
//
// Validated in tests against RollingMinMax(w) -> max - min as the
// composition reference.

#include <cmath>
#include <limits>
#include <vector>
#include "screamer/common/base.h"
#include "screamer/detail/block_extremum.h"
#include "screamer/detail/monotonic_deque.h"
#include <cstddef>

namespace screamer {

class RollingRange : public ScreamerBase {
public:
    explicit RollingRange(int window_size)
        : min_deque_(window_size), max_deque_(window_size) {}

    void reset() override {
        min_deque_.reset();
        max_deque_.reset();
    }

    // See RollingMax. Two block passes, one per extremum, still with no
    // data-dependent branching.
    void process_array_no_stride(double* y, const double* x, size_t size) override {
        if (min_deque_.samples_seen() != 0 || detail::has_nan(x, size)) {
            ScreamerBase::process_array_no_stride(y, x, size);
            return;
        }
        std::vector<double> lows(size);
        detail::block_extremum<false>(lows.data(), x, size, min_deque_.window_size());
        detail::block_extremum<true>(y, x, size, max_deque_.window_size());
        for (size_t i = 0; i < size; ++i) {
            y[i] -= lows[i];
        }
        const size_t window = static_cast<size_t>(min_deque_.window_size());
        for (size_t i = (size > window) ? size - window : 0; i < size; ++i) {
            min_deque_.append(x[i]);
            max_deque_.append(x[i]);
        }
    }

private:
    double process_scalar(double newValue) override {
        if (std::isnan(newValue)) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        const double current_min = min_deque_.append(newValue);
        const double current_max = max_deque_.append(newValue);
        return current_max - current_min;
    }

    detail::MinDeque min_deque_;
    detail::MaxDeque max_deque_;
};

}  // namespace screamer

#endif
