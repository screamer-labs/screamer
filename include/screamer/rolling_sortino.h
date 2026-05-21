#ifndef SCREAMER_ROLLING_SORTINO_H
#define SCREAMER_ROLLING_SORTINO_H

// RollingSortino: rolling Sortino ratio. Like Sharpe but the
// denominator is the *downside* deviation -- root mean square of
// returns below `target` (default 0):
//
//     Sortino[t] = sqrt(periods_per_year) * (mean - target) /
//                  sqrt(mean(min(r - target, 0)^2))
//
// Where mean and the downside-RMS are over the last `window_size`
// samples. Penalises only adverse moves (the Sharpe denominator
// includes both upside and downside variability).
//
// 1 -> 1 over a returns series. Composes a circular buffer + running
// sum and running sum-of-squared-downside-deviations. O(W) per step
// for the downside-RMS sweep.

#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>
#include "screamer/common/base.h"

namespace screamer {

class RollingSortino : public ScreamerBase {
public:
    RollingSortino(int window_size,
                   double periods_per_year = 1.0,
                   double target = 0.0)
        : window_size_(window_size),
          scale_(std::sqrt(periods_per_year)),
          target_(target)
    {
        if (window_size < 2) {
            throw std::invalid_argument("Window size must be at least 2.");
        }
        if (!(periods_per_year > 0.0)) {
            throw std::invalid_argument("periods_per_year must be positive.");
        }
        buffer_.resize(window_size_);
        reset();
    }

    void reset() override {
        std::fill(buffer_.begin(), buffer_.end(), 0.0);
        index_ = 0;
        size_ = 0;
        sum_ = 0.0;
    }

    double process_scalar(double r) override {
        // NaN policy "ignore": leave the buffer and sum alone.
        if (std::isnan(r)) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        const double oldValue = buffer_[index_];
        buffer_[index_] = r;
        index_++;
        if (index_ == window_size_) index_ = 0;
        if (size_ == window_size_) {
            sum_ += r - oldValue;
        } else {
            sum_ += r;
            size_++;
        }
        if (size_ < window_size_) {
            return std::numeric_limits<double>::quiet_NaN();
        }

        const double mean = sum_ / window_size_;

        // Downside RMS: mean of squared (r - target)^2 where r < target.
        double downside_sumsq = 0.0;
        for (int i = 0; i < window_size_; ++i) {
            const double d = buffer_[i] - target_;
            if (d < 0.0) downside_sumsq += d * d;
        }
        const double downside_rms = std::sqrt(downside_sumsq / window_size_);
        if (downside_rms <= 0.0) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        return scale_ * (mean - target_) / downside_rms;
    }

private:
    const int window_size_;
    const double scale_;
    const double target_;
    std::vector<double> buffer_;
    int index_ = 0;
    int size_ = 0;
    double sum_ = 0.0;
};

}  // namespace screamer

#endif
