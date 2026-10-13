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
#include <functional>
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
    // Strided inputs (a packed (T, 3) array, or a column of (T, K)) are read
    // in place, and a strided output is written in place. Declines when a bar
    // has a NaN, since under `ignore` a missing field skips a bar and the
    // block structure assumes every bar counts.
    bool process_columns(double* out, std::ptrdiff_t out_stride,
                         const std::array<double*, 3>& in,
                         const std::array<int64_t, 3>& in_stride,
                         const std::array<size_t, 3>& in_offset,
                         size_t out_offset, size_t size) override {
        const std::array<const double*, 3> x{in[0] + in_offset[0], in[1] + in_offset[1],
                                             in[2] + in_offset[2]};
        if (in_stride[0] == 1 && in_stride[1] == 1 && in_stride[2] == 1)
            return run_block<1>(out + out_offset, out_stride, x, in_stride, size);
        if (in_stride[0] == 3 && in_stride[1] == 3 && in_stride[2] == 3)
            return run_block<3>(out + out_offset, out_stride, x, in_stride, size);
        return run_block<0>(out + out_offset, out_stride, x, in_stride, size);
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
    // True when the three fields are the three adjacent values of one row,
    // in any order (addresses compared as integers: they may be separate
    // arrays).
    static bool packed_fields(const double* a, const double* b, const double* c) {
        const std::uintptr_t x = reinterpret_cast<std::uintptr_t>(a);
        const std::uintptr_t y = reinterpret_cast<std::uintptr_t>(b);
        const std::uintptr_t z = reinterpret_cast<std::uintptr_t>(c);
        const std::uintptr_t lo = std::min({x, y, z}), hi = std::max({x, y, z});
        const std::uintptr_t mid = x + y + z - lo - hi;
        return mid - lo == sizeof(double) && hi - mid == sizeof(double);
    }

    // process_columns on (high, low, close) at x, input k read with stride S
    // when S > 0 (1, or 3 for the fields of a packed (T, 3) array, known at
    // compile time) and xs[k] when S == 0, into y with stride y_stride.
    template <std::ptrdiff_t S>
    bool run_block(double* y, std::ptrdiff_t y_stride, const std::array<const double*, 3>& x,
                   const std::array<int64_t, 3>& xs, size_t size) {
        const std::ptrdiff_t sh = S > 0 ? S : xs[0];
        const std::ptrdiff_t sl = S > 0 ? S : xs[1];
        const std::ptrdiff_t sc = S > 0 ? S : xs[2];
        const double* high = x[0];
        const double* low = x[1];
        const double* close = x[2];
        bool nan;
        if (S == 1) {
            nan = detail::has_nan(high, size) || detail::has_nan(low, size) ||
                  detail::has_nan(close, size);
        } else if (S == 3 && packed_fields(high, low, close)) {
            // The rows of a packed (T, 3) array: its 3 * size values are one
            // contiguous block, checked in one pass.
            nan = detail::has_nan(std::min({high, low, close}, std::less<>()), 3 * size);
        } else {
            nan = detail::has_nan(high, size, sh) || detail::has_nan(low, size, sl) ||
                  detail::has_nan(close, size, sc);
        }
        if (nan) {
            return false;
        }

        const size_t warmup = static_cast<size_t>(window_size_) - 1;
        const double* const xs2[2] = {high, low};
        const std::ptrdiff_t ss[2] = {sh, sl};
        auto value = [&](size_t i, double h, double l) {
            if (i < warmup) return std::numeric_limits<double>::quiet_NaN();
            const double range = h - l;
            const double c = close[static_cast<std::ptrdiff_t>(i) * (S > 0 ? S : sc)];
            return (range <= 0.0) ? 0.0 : -100.0 * (h - c) / range;
        };
        if (S == 1) {
            // Separate arrays: the rolling high and low go to buffers, and the
            // value loop runs on contiguous arrays, which the compiler
            // vectorizes.
            double* highs = y;
            if (y_stride != 1) {
                highs_.resize(size);
                highs = highs_.data();
            }
            lows_.resize(size);
            double* const lows = lows_.data();
            double* const ys[2] = {highs, lows};
            detail::block_extrema<S, true, false>(xs2, ss, size, window_size_,
                                                  detail::ToArrays<2>{ys});
            for (size_t i = 0; i < size; ++i)
                y[static_cast<std::ptrdiff_t>(i) * y_stride] = value(i, highs[i], lows[i]);
        } else {
            // Strided fields: the value of row i is made as the block pass
            // reads row i, so close comes from the cache lines just loaded.
            struct Fused {
                const decltype(value)& value;
                double* y;
                std::ptrdiff_t y_stride;
                double e[2];
                void put(size_t k, size_t, double v) { e[k] = v; }
                void row(size_t i) {
                    y[static_cast<std::ptrdiff_t>(i) * y_stride] = value(i, e[0], e[1]);
                }
            };
            detail::block_extrema<S, true, false>(xs2, ss, size, window_size_,
                                                  Fused{value, y, y_stride, {}});
        }

        // Leave state where the per-sample loop would have left it.
        for (size_t i = (size > static_cast<size_t>(window_size_))
                            ? size - static_cast<size_t>(window_size_) : 0;
             i < size; ++i) {
            max_deque_.append(high[(std::ptrdiff_t) i * sh]);
            min_deque_.append(low[(std::ptrdiff_t) i * sl]);
        }
        n_seen_ = static_cast<int>(std::min<size_t>(size, static_cast<size_t>(window_size_)));
        return true;
    }

    const int window_size_;
    // Batch scratch, reused across columns and calls: rolling lows, and
    // rolling highs when the output is strided.
    std::vector<double> lows_, highs_;
    detail::MaxDeque max_deque_;
    detail::MinDeque min_deque_;
    int n_seen_ = 0;
};

}  // namespace screamer

#endif
