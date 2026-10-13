#ifndef SCREAMER_DETAIL_BLOCK_EXTREMUM_H
#define SCREAMER_DETAIL_BLOCK_EXTREMUM_H

// block_extremum: sliding-window min/max over a whole array, without any
// data-dependent branching.
//
// The streaming operators use a monotonic deque, which is comparison-optimal
// and O(1) per sample in the worst case. Its cost on real data is not the
// comparisons but the pop loop, whose trip count depends on the data: on a
// random series that branch mispredicts on most samples, which measured as
// ~6 ns/sample of pipeline flush (RollingMax(50) ran at 8.6 ns/sample on
// random input against 2.4 on monotone input, where the loop is predictable).
//
// This is the block decomposition instead. Split the series into blocks of
// `window`, take a running maximum forward through each block and another
// backward, and any window of length `window` spans two adjacent blocks:
//
//     y[i] = max( suffix[i - window + 1], prefix[i] )
//
// Two comparisons per element, no data-dependent branch, and the cost does
// not depend on the data at all.
//
// It cannot replace the deque in the streaming engine: the backward pass
// means the output for sample i needs samples up to the end of i's block, so
// up to `window` samples of lookahead. That is free in batch and impossible
// on a live feed. Hence both, with the array path delegating here and the
// event path keeping the deque.
//
// Callers must exclude NaN before calling: under the `ignore` policy a NaN
// does not enter the window, which would break the fixed block structure.

#include <algorithm>
#include <cstddef>
#include <utility>
#include <vector>

namespace screamer {
namespace detail {

// True if any sample is NaN. Callers use this to decide whether the block
// form applies: under the `ignore` policy a NaN does not enter the window,
// which the fixed block structure cannot express.
inline bool has_nan(const double* x, std::size_t size) {
    for (std::size_t i = 0; i < size; ++i) {
        if (x[i] != x[i]) {
            return true;
        }
    }
    return false;
}

// The same for size samples x[i * stride].
inline bool has_nan(const double* x, std::size_t size, std::ptrdiff_t stride) {
    for (std::size_t i = 0; i < size; ++i) {
        const double v = x[static_cast<std::ptrdiff_t>(i) * stride];
        if (v != v) {
            return true;
        }
    }
    return false;
}


// The block decomposition for K series at once: out.put(k, i, e) receives
// e = extremum of x[k][max(0, i - window + 1) .. i], a max where IsMax[k] and
// a min otherwise, and out.row(i) follows the K puts of row i, for i = 0, 1,
// ..., matching the deque's output during warmup as well as afterwards. Both
// run while sample i is being read, so a caller that combines the extrema
// with other fields of row i reads them from the cache lines just loaded
// (ToArrays stores them instead). The series
// are stepped together, which overlaps their dependency chains. Sample i of
// series k is x[k][i * s], with s = S when S > 0 (a stride known at compile
// time, such as 3 for a field of a packed (T, 3) array, which the compiler
// reads as one row) and s = stride[k] when S == 0.
template <std::ptrdiff_t S, bool... IsMax, class Out>
void block_extrema(const double* const* x, const std::ptrdiff_t* stride, std::size_t size,
                   int window, Out&& out) {
    constexpr std::size_t K = sizeof...(IsMax);
    constexpr bool kMax[K] = {IsMax...};
    const std::size_t w = static_cast<std::size_t>(window);
    if (size == 0) {
        return;
    }
    std::ptrdiff_t s[K];
    for (std::size_t k = 0; k < K; ++k) s[k] = S > 0 ? S : stride[k];
    auto at = [&](std::size_t k, std::size_t i) {
        return x[k][static_cast<std::ptrdiff_t>(i) * (S > 0 ? S : s[k])];
    };
    auto better = [](std::size_t k, double a, double b) {
        return kMax[k] ? (a > b ? a : b) : (a < b ? a : b);
    };

    // Suffix extrema of the current block and of the previous one. The window
    // for sample i reaches back into the previous block, so both are live.
    std::vector<double> scratch(2 * K * w);
    double* cur[K];
    double* prev[K];
    for (std::size_t k = 0; k < K; ++k) {
        cur[k] = scratch.data() + (2 * k) * w;
        prev[k] = scratch.data() + (2 * k + 1) * w;
    }

    for (std::size_t start = 0; start < size; start += w) {
        const std::size_t end = std::min(start + w, size);
        const std::size_t length = end - start;

        // Backward pass: cur[k][j] is the extremum of x[k][start+j .. end-1].
        double running[K];
        for (std::size_t k = 0; k < K; ++k) {
            running[k] = at(k, end - 1);
            cur[k][length - 1] = running[k];
        }
        for (std::size_t j = length - 1; j-- > 0;) {
            for (std::size_t k = 0; k < K; ++k) {
                running[k] = better(k, running[k], at(k, start + j));
                cur[k][j] = running[k];
            }
        }

        // Forward pass: prefix is the extremum of x[k][start .. i].
        double prefix[K];
        for (std::size_t j = 0; j < length; ++j) {
            const std::size_t i = start + j;
            for (std::size_t k = 0; k < K; ++k) {
                prefix[k] = (j == 0) ? at(k, i) : better(k, prefix[k], at(k, i));
                if (i + 1 < w) {
                    // Warmup: the window is x[0 .. i], which is this block's
                    // prefix because a short window cannot have crossed a
                    // block boundary.
                    out.put(k, i, prefix[k]);
                } else {
                    const std::size_t low = i + 1 - w;
                    out.put(k, i, (low >= start)
                        ? better(k, cur[k][low - start], prefix[k])
                        : better(k, prev[k][low - (start - w)], prefix[k]));
                }
            }
            out.row(i);
        }

        for (std::size_t k = 0; k < K; ++k) std::swap(cur[k], prev[k]);
    }
}

// The output of block_extrema into K arrays: y[k][i].
template <std::size_t K>
struct ToArrays {
    double* const* y;
    void put(std::size_t k, std::size_t i, double e) const { y[k][i] = e; }
    void row(std::size_t) const {}
};

// IsMax selects max (true) or min (false): y[i] = extremum of
// x[max(0, i - window + 1) .. i] (block_extrema of one contiguous series).
template <bool IsMax>
void block_extremum(double* y, const double* x, std::size_t size, int window) {
    block_extrema<1, IsMax>(&x, nullptr, size, window, ToArrays<1>{&y});
}

// Same decomposition, returning the *position* of the extremum within the
// window rather than its value: 0 is the oldest sample in the window.
//
// Ties must break the way the deque does, or the two paths disagree on flat
// stretches. MonotonicDeque<true> pops while `back <= value`, so an equal
// later sample displaces an earlier one and the newest index wins; the same
// holds for min. So the forward scan replaces on `>=` (later wins), the
// backward scan replaces only on `>` (so scanning right to left keeps the
// later index), and a tie between the two halves goes to the prefix, which
// is the later part of the window.
template <bool IsMax>
void block_arg_extremum(double* y, const double* x, std::size_t size, int window) {
    const std::size_t w = static_cast<std::size_t>(window);
    if (size == 0) {
        return;
    }

    auto strictly_better = [](double a, double b) { return IsMax ? (a > b) : (a < b); };
    auto at_least = [](double a, double b) { return IsMax ? (a >= b) : (a <= b); };

    std::vector<double> suffix_value_current(w), suffix_value_previous(w);
    std::vector<std::size_t> suffix_index_current(w), suffix_index_previous(w);

    for (std::size_t start = 0; start < size; start += w) {
        const std::size_t end = std::min(start + w, size);
        const std::size_t length = end - start;

        double best_value = x[end - 1];
        std::size_t best_index = end - 1;
        suffix_value_current[length - 1] = best_value;
        suffix_index_current[length - 1] = best_index;
        for (std::size_t j = length - 1; j-- > 0;) {
            const std::size_t i = start + j;
            if (strictly_better(x[i], best_value)) {   // strict: keep the later index on ties
                best_value = x[i];
                best_index = i;
            }
            suffix_value_current[j] = best_value;
            suffix_index_current[j] = best_index;
        }

        double prefix_value = 0.0;
        std::size_t prefix_index = 0;
        for (std::size_t j = 0; j < length; ++j) {
            const std::size_t i = start + j;
            if (j == 0 || at_least(x[i], prefix_value)) {   // >=: a later equal wins
                prefix_value = x[i];
                prefix_index = i;
            }

            std::size_t chosen;
            if (i + 1 < w) {
                chosen = prefix_index;
            } else {
                const std::size_t low = i + 1 - w;
                const bool in_this_block = (low >= start);
                const double other_value = in_this_block
                    ? suffix_value_current[low - start]
                    : suffix_value_previous[low - (start - w)];
                const std::size_t other_index = in_this_block
                    ? suffix_index_current[low - start]
                    : suffix_index_previous[low - (start - w)];
                // Tie goes to the prefix half, which holds the later index.
                chosen = strictly_better(other_value, prefix_value) ? other_index : prefix_index;
            }

            const std::size_t window_start = (i + 1 < w) ? 0 : (i + 1 - w);
            y[i] = static_cast<double>(chosen - window_start);
        }

        suffix_value_previous.swap(suffix_value_current);
        suffix_index_previous.swap(suffix_index_current);
    }
}

}  // namespace detail
}  // namespace screamer

#endif  // SCREAMER_DETAIL_BLOCK_EXTREMUM_H
