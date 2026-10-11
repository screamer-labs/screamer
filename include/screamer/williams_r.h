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

#include <array>
#include <vector>
#include <algorithm>
#include <limits>
#include <stdexcept>
#include "screamer/common/functor_base.h"
#include "screamer/common/float_info.h"
#include "screamer/detail/block_extremum.h"
#include "screamer/detail/monotonic_deque.h"
#include <cstddef>
#include <cstdint>

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
    // Strided inputs (a packed (T, 3) array, or a column of (T, K)) are first
    // gathered into contiguous scratch, and a strided output is written in
    // place. Declines when a bar has a NaN, since under `ignore` a missing
    // field skips a bar and the block structure assumes every bar counts.
    bool process_columns(double* out, std::ptrdiff_t out_stride,
                         const std::array<double*, 3>& in,
                         const std::array<int64_t, 3>& in_stride,
                         const std::array<size_t, 3>& in_offset,
                         size_t out_offset, size_t size) override {
        // Contiguous view of input k: the input itself when its stride is 1,
        // otherwise a gather into scratch_[k].
        std::array<const double*, 3> x;
        for (size_t k = 0; k < 3; ++k) {
            const double* src = in[k] + in_offset[k];
            if (in_stride[k] == 1) {
                x[k] = src;
                continue;
            }
            scratch_[k].resize(size);
            for (size_t i = 0; i < size; ++i) scratch_[k][i] = src[(std::ptrdiff_t) i * in_stride[k]];
            x[k] = scratch_[k].data();
        }
        const double* high = x[0];
        const double* low = x[1];
        const double* close = x[2];
        if (detail::has_nan(high, size) || detail::has_nan(low, size) ||
            detail::has_nan(close, size)) {
            return false;
        }

        // Rolling high goes straight into a unit-stride output, else into scratch.
        double* y = out + out_offset;
        double* highs = y;
        if (out_stride != 1) {
            highs_.resize(size);
            highs = highs_.data();
        }
        lows_.resize(size);
        detail::block_extremum<true>(highs, high, size, window_size_);
        detail::block_extremum<false>(lows_.data(), low, size, window_size_);

        const size_t warmup = static_cast<size_t>(window_size_) - 1;
        for (size_t i = 0; i < size; ++i) {
            double r;
            if (i < warmup) {
                r = std::numeric_limits<double>::quiet_NaN();
            } else {
                const double range = highs[i] - lows_[i];
                r = (range <= 0.0) ? 0.0 : -100.0 * (highs[i] - close[i]) / range;
            }
            y[(std::ptrdiff_t) i * out_stride] = r;
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
    // Batch scratch, reused across columns and calls: gathered strided inputs,
    // rolling lows, and rolling highs when the output is strided.
    std::array<std::vector<double>, 3> scratch_;
    std::vector<double> lows_, highs_;
};

}  // namespace screamer

#endif
