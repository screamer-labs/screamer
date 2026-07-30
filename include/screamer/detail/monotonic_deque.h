#ifndef SCREAMER_DETAIL_MONOTONIC_DEQUE_H
#define SCREAMER_DETAIL_MONOTONIC_DEQUE_H

// MonotonicDeque<IsMax>: amortised O(1) sliding-window extremum.
//
// Holds (value, absolute_sample_index) pairs. The deque is kept
// monotonic (non-increasing for max, non-decreasing for min) by popping
// from the back whenever a new value would invalidate trailing
// candidates. The front is always the current rolling extremum, and
// its absolute_sample_index lets callers compute argmin/argmax.
//
// Used by RollingMin, RollingMax, RollingMinMax, RollingArgmin,
// RollingArgmax, RollingRange, WilliamsR, Stoch, StochRSI and
// DonchianChannels.
//
// Storage is a fixed ring buffer, sized once at construction. It never
// holds more than window_size live entries (older ones have expired), and
// one extra slot absorbs the moment between pushing a new candidate and
// dropping the expired front, so window_size + 1 is enough and nothing is
// allocated after construction.
//
// This used to be a std::deque, which is what made the whole family slow:
// std::deque allocates in chunks and reaches its elements through a map of
// block pointers, so every push, pop and front() costs an indirection.
// RollingMax(50) measured 9.9 ns/sample against TA-Lib's MAX at 1.1, and
// flat in window size, so the gap was constant-factor rather than
// algorithmic. Values and indices live in separate arrays because the pop
// loop only ever compares values.

#include <cstddef>
#include <stdexcept>
#include <vector>

namespace screamer::detail {

template <bool IsMax>
class MonotonicDeque {
public:
    explicit MonotonicDeque(int window_size)
        : window_size_(window_size),
          capacity_(window_size > 0 ? static_cast<std::size_t>(window_size) + 1 : 1),
          values_(capacity_),
          indices_(capacity_)
    {
        if (window_size <= 0) {
            throw std::invalid_argument("Window size must be positive.");
        }
    }

    void reset() {
        head_ = 0;
        count_ = 0;
        index_ = 0;
    }

    // Append a new sample and return the current rolling extremum.
    double append(double value) {
        // Drop trailing candidates the new value dominates: they can never
        // be the extremum again while it stays in the window.
        while (count_ > 0) {
            std::size_t back = head_ + count_ - 1;
            if (back >= capacity_) {
                back -= capacity_;
            }
            if constexpr (IsMax) {
                if (values_[back] > value) {
                    break;
                }
            } else {
                if (values_[back] < value) {
                    break;
                }
            }
            --count_;
        }

        std::size_t slot = head_ + count_;
        if (slot >= capacity_) {
            slot -= capacity_;
        }
        values_[slot] = value;
        indices_[slot] = index_;
        ++count_;

        // Expire the front if it has fallen out of the window.
        if (indices_[head_] <= index_ - window_size_) {
            ++head_;
            if (head_ == capacity_) {
                head_ = 0;
            }
            --count_;
        }

        ++index_;
        return values_[head_];
    }

    // Current extremum value. Undefined if append() has never been called.
    double front_value() const { return values_[head_]; }

    // Absolute sample index (0-based, since the start of the stream / last
    // reset()) of the current extremum.
    int front_absolute_index() const { return indices_[head_]; }

    // Offset of the current extremum within the active window:
    //   0  = oldest sample currently in the window
    //   W-1 (or n-1 during warmup) = newest sample
    // Matches numpy.argmin / numpy.argmax of the window slice.
    int front_window_offset() const {
        const int window_start = (index_ < window_size_) ? 0 : (index_ - window_size_);
        return indices_[head_] - window_start;
    }

    // Number of samples currently in the active window
    // (= min(samples_seen, window_size)).
    int current_size() const {
        return (index_ < window_size_) ? index_ : window_size_;
    }

    int samples_seen() const { return index_; }
    int window_size() const { return window_size_; }

private:
    int window_size_;
    std::size_t capacity_;
    std::vector<double> values_;
    std::vector<int> indices_;
    std::size_t head_ = 0;   // slot holding the current extremum
    std::size_t count_ = 0;  // live candidates
    int index_ = 0;          // samples appended since construction / reset
};

using MaxDeque = MonotonicDeque<true>;
using MinDeque = MonotonicDeque<false>;

}  // namespace screamer::detail

#endif  // SCREAMER_DETAIL_MONOTONIC_DEQUE_H
