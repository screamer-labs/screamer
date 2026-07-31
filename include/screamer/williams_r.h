#ifndef SCREAMER_WILLIAMS_R_H
#define SCREAMER_WILLIAMS_R_H

// WilliamsR: Williams %R (Larry Williams 1973). Normalised position of
// the close within the recent (high, low) range, scaled to [-100, 0]:
// 0 means the close is at the period high, -100 means it is at the
// period low.
//
//     %R[t] = -100 * (high_n - close) / (high_n - low_n)
//
// where high_n / low_n are the rolling max / min of high and low over
// the period. A 3 -> 1 functor (FunctorBase<_, 3, 1>): inputs are
// (high, low, close) in TA-Lib's argument order.
//
// Composition: two detail::MonotonicDeque instances (the same
// primitive RollingMin / RollingMax / RollingArgmin / RollingArgmax /
// RollingRange use). Amortised O(1) per step.
//
// Warmup: NaN for the first window_size - 1 samples; first valid
// output at sample index window_size - 1 (TA-Lib's convention).
//
// Range-zero handling: when high_n == low_n (a perfectly flat
// segment), the formula is mathematically undefined. TA-Lib returns 0
// in that case; we follow.

#include <vector>
#include <algorithm>
#include <limits>
#include <stdexcept>
#include "screamer/common/functor_base.h"
#include "screamer/common/float_info.h"
#include "screamer/detail/block_extremum.h"
#include "screamer/detail/monotonic_deque.h"

namespace screamer {

class WilliamsR : public FunctorBase<WilliamsR, 3, 1> {
public:
    explicit WilliamsR(int window_size = 14)
        : window_size_(window_size),
          max_deque_(window_size),
          min_deque_(window_size)
    {
        if (window_size < 1) {
            throw std::invalid_argument("Window size must be at least 1.");
        }
    }

    void reset() override {
        max_deque_.reset();
        min_deque_.reset();
        n_seen_ = 0;
    }

    // Batch path: the rolling high and low come from the block decomposition,
    // which has no data-dependent branching, and the rest is elementwise. The
    // event path keeps the deques. See detail/block_extremum.h.
    //
    // Declines unless the three inputs and the output are contiguous and the
    // bars are free of NaN, since under `ignore` a missing field skips a bar
    // and the block structure assumes every bar counts.
    bool process_columns(double* out, std::ptrdiff_t out_stride,
                         const std::array<double*, 3>& in,
                         const std::array<int64_t, 3>& in_stride,
                         const std::array<size_t, 3>& in_offset,
                         size_t out_offset, size_t size) override {
        if (out_stride != 1 || in_stride[0] != 1 || in_stride[1] != 1 || in_stride[2] != 1) {
            return false;
        }
        const double* high = in[0] + in_offset[0];
        const double* low = in[1] + in_offset[1];
        const double* close = in[2] + in_offset[2];
        if (detail::has_nan(high, size) || detail::has_nan(low, size) ||
            detail::has_nan(close, size)) {
            return false;
        }

        double* y = out + out_offset;
        std::vector<double> lows(size);
        detail::block_extremum<true>(y, high, size, window_size_);
        detail::block_extremum<false>(lows.data(), low, size, window_size_);

        const size_t warmup = static_cast<size_t>(window_size_) - 1;
        for (size_t i = 0; i < size; ++i) {
            if (i < warmup) {
                y[i] = std::numeric_limits<double>::quiet_NaN();
                continue;
            }
            const double range = y[i] - lows[i];
            y[i] = (range <= 0.0) ? 0.0 : -100.0 * (y[i] - close[i]) / range;
        }

        // Leave state where the per-sample loop would have left it.
        for (size_t i = (size > static_cast<size_t>(window_size_))
                            ? size - static_cast<size_t>(window_size_) : 0;
             i < size; ++i) {
            max_deque_.append(high[i]);
            min_deque_.append(low[i]);
        }
        n_seen_ = static_cast<int>(std::min<size_t>(size, static_cast<size_t>(window_size_)));
        return true;
    }

    ResultTuple call(const InputArray& inputs) override {
        const double high  = inputs[0];
        const double low   = inputs[1];
        const double close = inputs[2];
        // nan_policy: ignore. A bar with any missing field is skipped
        // whole: nothing is stored and warmup does not advance.
        if (any_nan(high, low, close)) {
            return std::numeric_limits<double>::quiet_NaN();
        }

        const double high_n = max_deque_.append(high);
        const double low_n  = min_deque_.append(low);

        if (n_seen_ < window_size_) {
            n_seen_++;
        }
        if (n_seen_ < window_size_) {
            return std::numeric_limits<double>::quiet_NaN();
        }

        const double range = high_n - low_n;
        if (range <= 0.0) {
            return 0.0;
        }
        return -100.0 * (high_n - close) / range;
    }

private:
    const int window_size_;
    detail::MaxDeque max_deque_;
    detail::MinDeque min_deque_;
    int n_seen_ = 0;
};

}  // namespace screamer

#endif
