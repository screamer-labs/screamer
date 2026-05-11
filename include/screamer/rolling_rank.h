#ifndef SCREAMER_ROLLING_RANK_H
#define SCREAMER_ROLLING_RANK_H

// RollingRank: rank of the most recent value within the trailing
// window, using pandas's "average" tie-breaking rule.
//
//     rank[t] = average rank position of x[t] within
//               sort(x[t-w+1..t]) where ties get the mean rank.
//
// Output is a 1-based rank in [1, w]. For the [0, 1] percentile
// form see RollingPercentile.
//
// 1 -> 1. Circular window buffer + a per-step sweep to count <=
// and == ties; O(W) per step. Matches pandas.Series.rolling(w).rank().

#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>
#include "screamer/common/base.h"

namespace screamer {

class RollingRank : public ScreamerBase {
public:
    explicit RollingRank(int window_size) : window_size_(window_size) {
        if (window_size < 1) {
            throw std::invalid_argument("Window size must be positive.");
        }
        buffer_.resize(window_size_);
        reset();
    }

    void reset() override {
        std::fill(buffer_.begin(), buffer_.end(), 0.0);
        index_ = 0;
        size_ = 0;
    }

    double process_scalar(double x) override {
        buffer_[index_] = x;
        index_++;
        if (index_ == window_size_) index_ = 0;
        if (size_ < window_size_) {
            size_++;
            if (size_ < window_size_) {
                return std::numeric_limits<double>::quiet_NaN();
            }
        }
        // Count < x and == x within the buffer.
        int less = 0;
        int equal = 0;
        for (int i = 0; i < window_size_; ++i) {
            if (buffer_[i] < x)        less++;
            else if (buffer_[i] == x)  equal++;
        }
        // pandas "average" tie rule: rank = (number of <= values) -
        // (ties - 1) / 2, expressed as 1-based.
        return (less + 1) + (equal - 1) * 0.5;
    }

private:
    const int window_size_;
    std::vector<double> buffer_;
    int index_ = 0;
    int size_ = 0;
};

}  // namespace screamer

#endif
