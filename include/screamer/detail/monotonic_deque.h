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
// RollingArgmax, and RollingRange.

#include <cstddef>
#include <deque>
#include <stdexcept>
#include <utility>

namespace screamer::detail {

template <bool IsMax>
class MonotonicDeque {
public:
    explicit MonotonicDeque(int window_size) : window_size_(window_size) {
        if (window_size <= 0) {
            throw std::invalid_argument("Window size must be positive.");
        }
    }

    void reset() {
        deque_.clear();
        index_ = 0;
    }

    // Append a new sample and return the current rolling extremum.
    double append(double value) {
        if constexpr (IsMax) {
            while (!deque_.empty() && deque_.back().first <= value) {
                deque_.pop_back();
            }
        } else {
            while (!deque_.empty() && deque_.back().first >= value) {
                deque_.pop_back();
            }
        }
        deque_.emplace_back(value, index_);
        if (deque_.front().second <= index_ - window_size_) {
            deque_.pop_front();
        }
        index_++;
        return deque_.front().first;
    }

    // Current extremum value. Undefined if append() has never been called.
    double front_value() const { return deque_.front().first; }

    // Absolute sample index (0-based, since the start of the stream / last
    // reset()) of the current extremum.
    int front_absolute_index() const { return deque_.front().second; }

    // Offset of the current extremum within the active window:
    //   0  = oldest sample currently in the window
    //   W-1 (or n-1 during warmup) = newest sample
    // Matches numpy.argmin / numpy.argmax of the window slice.
    int front_window_offset() const {
        const int window_start = (index_ < window_size_) ? 0 : (index_ - window_size_);
        return deque_.front().second - window_start;
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
    int index_ = 0;
    std::deque<std::pair<double, int>> deque_;
};

using MaxDeque = MonotonicDeque<true>;
using MinDeque = MonotonicDeque<false>;

}  // namespace screamer::detail

#endif  // SCREAMER_DETAIL_MONOTONIC_DEQUE_H
