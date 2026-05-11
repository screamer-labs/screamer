#ifndef SCREAMER_ROLLING_MAX_DRAWDOWN_H
#define SCREAMER_ROLLING_MAX_DRAWDOWN_H

// RollingMaxDrawdown: the worst peak-to-trough drawdown experienced
// within the most recent `window_size` bars.
//
// At each step:
//   - keep a circular buffer of the last w price samples
//   - sweep the buffer in time order, tracking the running peak
//     and the worst (most negative) drawdown observed from that peak
//
// 1 -> 1. Output is in (-1, 0]. First valid at sample index w-1.
// O(window_size) per step, matching the standard definition. For a
// strictly cheaper "current drawdown vs rolling peak" approximation
// users can compose RollingMax themselves:
//
//     rolling_dd = price / RollingMax(window)(price) - 1

#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>
#include <algorithm>
#include "screamer/common/base.h"

namespace screamer {

class RollingMaxDrawdown : public ScreamerBase {
public:
    explicit RollingMaxDrawdown(int window_size)
        : window_size_(window_size)
    {
        if (window_size < 2) {
            throw std::invalid_argument("Window size must be at least 2.");
        }
        buffer_.resize(window_size_);
        reset();
    }

    void reset() override {
        std::fill(buffer_.begin(), buffer_.end(), 0.0);
        index_ = 0;
        size_ = 0;
    }

    double process_scalar(double price) override {
        buffer_[index_] = price;
        index_++;
        if (index_ == window_size_) index_ = 0;
        if (size_ < window_size_) {
            size_++;
            if (size_ < window_size_) {
                return std::numeric_limits<double>::quiet_NaN();
            }
        }

        // After buffer is full: walk the window in time order. The
        // oldest sample lives at the current `index_` (next write
        // slot), so we iterate (index_, index_+1, ..., index_+w-1) mod w.
        double peak = -std::numeric_limits<double>::infinity();
        double worst = 0.0;
        for (int k = 0; k < window_size_; ++k) {
            const int i = (index_ + k) % window_size_;
            const double p = buffer_[i];
            if (p > peak) peak = p;
            if (peak > 0.0) {
                const double dd = p / peak - 1.0;
                if (dd < worst) worst = dd;
            }
        }
        return worst;
    }

private:
    const int window_size_;
    std::vector<double> buffer_;
    int index_ = 0;
    int size_ = 0;
};

}  // namespace screamer

#endif
