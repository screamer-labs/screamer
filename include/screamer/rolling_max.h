#ifndef SCREAMER_ROLLING_MAX_H
#define SCREAMER_ROLLING_MAX_H

#include <cmath>
#include <cstddef>
#include <limits>
#include "screamer/common/base.h"
#include "screamer/detail/block_extremum.h"
#include "screamer/detail/monotonic_deque.h"

namespace screamer {

class RollingMax : public ScreamerBase {
public:
    explicit RollingMax(int window_size) : deque_(window_size) {}

    void reset() override { deque_.reset(); }

    // Array path. The event path keeps the monotonic deque, which needs no
    // lookahead; over a whole array the block decomposition computes the same
    // values with no data-dependent branching, which is what the deque's cost
    // actually is on random input. See detail/block_extremum.h.
    //
    // Two conditions must hold, and both fall back to the scalar loop rather
    // than change any result:
    //   * the operator is unused, since the block form starts its window at
    //     index 0 and cannot continue a window left over from earlier calls;
    //   * the input is free of NaN, since under the `ignore` policy a NaN does
    //     not enter the window, and the block structure assumes every sample
    //     does.
    void process_array_no_stride(double* y, const double* x, size_t size) override {
        if (deque_.samples_seen() != 0) {
            ScreamerBase::process_array_no_stride(y, x, size);
            return;
        }
        for (size_t i = 0; i < size; ++i) {
            if (std::isnan(x[i])) {
                ScreamerBase::process_array_no_stride(y, x, size);
                return;
            }
        }
        detail::block_extremum<true>(y, x, size, deque_.window_size());
        // Leave the deque holding what the scalar loop would have left, so a
        // later scalar call continues correctly. The eager API resets after
        // this returns; this keeps the two paths interchangeable regardless.
        const size_t window = static_cast<size_t>(deque_.window_size());
        const size_t tail = (size > window) ? size - window : 0;
        for (size_t i = tail; i < size; ++i) {
            deque_.append(x[i]);
        }
    }

private:
    double process_scalar(double newValue) override {
        // NaN policy "ignore": leave the deque untouched and emit NaN.
        if (std::isnan(newValue)) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        return deque_.append(newValue);
    }

    detail::MaxDeque deque_;
};

}  // namespace screamer

#endif
