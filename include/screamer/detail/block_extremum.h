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


// IsMax selects max (true) or min (false).
// y[i] = extremum of x[max(0, i - window + 1) .. i], matching the deque's
// output during warmup as well as afterwards.
template <bool IsMax>
void block_extremum(double* y, const double* x, std::size_t size, int window) {
    const std::size_t w = static_cast<std::size_t>(window);
    if (size == 0) {
        return;
    }

    auto better = [](double a, double b) { return IsMax ? (a > b ? a : b) : (a < b ? a : b); };

    // Suffix maxima of the current block and of the previous one. The window
    // for sample i reaches back into the previous block, so both are live.
    std::vector<double> suffix_current(w);
    std::vector<double> suffix_previous(w);

    for (std::size_t start = 0; start < size; start += w) {
        const std::size_t end = std::min(start + w, size);
        const std::size_t length = end - start;

        // Backward pass: suffix_current[j] is the extremum of x[start+j .. end-1].
        double running = x[end - 1];
        suffix_current[length - 1] = running;
        for (std::size_t j = length - 1; j-- > 0;) {
            running = better(running, x[start + j]);
            suffix_current[j] = running;
        }

        // Forward pass: prefix is the extremum of x[start .. i].
        double prefix = x[start];
        for (std::size_t j = 0; j < length; ++j) {
            const std::size_t i = start + j;
            prefix = (j == 0) ? x[i] : better(prefix, x[i]);

            if (i + 1 < w) {
                // Warmup: the window is x[0 .. i], which is this block's prefix
                // because a short window cannot have crossed a block boundary.
                y[i] = prefix;
            } else {
                const std::size_t low = i + 1 - w;
                y[i] = (low >= start)
                    ? better(suffix_current[low - start], prefix)
                    : better(suffix_previous[low - (start - w)], prefix);
            }
        }

        suffix_previous.swap(suffix_current);
    }
}

// block_extremum<true> of xh into ymax and block_extremum<false> of xl into
// ymin, in one pass over the blocks: the same values, with each pair of
// inputs read together (the high and low fields of one row of a packed array
// share a cache line). With Strided, sample i of xh is xh[i * sh], and of xl
// xl[i * sl].
template <bool Strided = false>
void block_max_min(double* ymax, double* ymin, const double* xh, const double* xl,
                   std::size_t size, int window, std::ptrdiff_t sh = 1, std::ptrdiff_t sl = 1) {
    const std::size_t w = static_cast<std::size_t>(window);
    const std::ptrdiff_t ah = Strided ? sh : 1, al = Strided ? sl : 1;
    auto hi = [xh, ah](std::size_t i) { return xh[static_cast<std::ptrdiff_t>(i) * ah]; };
    auto lo = [xl, al](std::size_t i) { return xl[static_cast<std::ptrdiff_t>(i) * al]; };
    if (size == 0) {
        return;
    }
    auto max = [](double a, double b) { return a > b ? a : b; };
    auto min = [](double a, double b) { return a < b ? a : b; };

    // As block_extremum: the suffix extrema of the current block and the
    // previous one, for each input.
    std::vector<double> hcur(w), hprev(w), lcur(w), lprev(w);
    for (std::size_t start = 0; start < size; start += w) {
        const std::size_t end = std::min(start + w, size);
        const std::size_t length = end - start;

        double rh = hi(end - 1), rl = lo(end - 1);
        hcur[length - 1] = rh;
        lcur[length - 1] = rl;
        for (std::size_t j = length - 1; j-- > 0;) {
            rh = max(rh, hi(start + j));
            rl = min(rl, lo(start + j));
            hcur[j] = rh;
            lcur[j] = rl;
        }

        double ph = hi(start), pl = lo(start);
        for (std::size_t j = 0; j < length; ++j) {
            const std::size_t i = start + j;
            ph = (j == 0) ? hi(i) : max(ph, hi(i));
            pl = (j == 0) ? lo(i) : min(pl, lo(i));
            if (i + 1 < w) {
                ymax[i] = ph;
                ymin[i] = pl;
            } else {
                const std::size_t low = i + 1 - w;
                if (low >= start) {
                    ymax[i] = max(hcur[low - start], ph);
                    ymin[i] = min(lcur[low - start], pl);
                } else {
                    ymax[i] = max(hprev[low - (start - w)], ph);
                    ymin[i] = min(lprev[low - (start - w)], pl);
                }
            }
        }
        hprev.swap(hcur);
        lprev.swap(lcur);
    }
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
