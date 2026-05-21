#ifndef SCREAMER_ROLLING_PERCENTILE_H
#define SCREAMER_ROLLING_PERCENTILE_H

// RollingPercentile: where the most recent value sits inside the
// trailing window, expressed as a fraction in [0, 1].
//
//     percentile[t] = (RollingRank[t] - 1) / (w - 1)
//
// Effectively the inverse of RollingQuantile: instead of "give me
// the value at the 0.75 quantile" you ask "what quantile is my
// current value at". Same algorithm and complexity as RollingRank.
// Matches pandas.Series.rolling(w).rank(pct=True) -- by pandas's
// convention `rank(pct=True)` divides by w (not w-1).
//
// We match pandas exactly: percentile = RollingRank / w. So output
// is in [1/w, 1].

#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>
#include "screamer/common/base.h"

namespace screamer {

class RollingPercentile : public ScreamerBase {
public:
    explicit RollingPercentile(int window_size) : window_size_(window_size) {
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
        // NaN policy "ignore": leave the buffer untouched, emit NaN.
        if (std::isnan(x)) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        buffer_[index_] = x;
        index_++;
        if (index_ == window_size_) index_ = 0;
        if (size_ < window_size_) {
            size_++;
            if (size_ < window_size_) {
                return std::numeric_limits<double>::quiet_NaN();
            }
        }
        int less = 0;
        int equal = 0;
        for (int i = 0; i < window_size_; ++i) {
            if (buffer_[i] < x)       less++;
            else if (buffer_[i] == x) equal++;
        }
        const double avg_rank = (less + 1) + (equal - 1) * 0.5;
        return avg_rank / window_size_;
    }

private:
    const int window_size_;
    std::vector<double> buffer_;
    int index_ = 0;
    int size_ = 0;
};

}  // namespace screamer

#endif
