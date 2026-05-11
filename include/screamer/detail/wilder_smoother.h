#ifndef SCREAMER_DETAIL_WILDER_SMOOTHER_H
#define SCREAMER_DETAIL_WILDER_SMOOTHER_H

// WilderSmoother: J. Welles Wilder's recursive smoother (1978). Used
// by ATR, RSI's Wilder mode, ADX, and DI/DM smoothing.
//
//     smoothed[w] = (1/w) * sum_{i=1..w} x[i]              (SMA seed)
//     smoothed[t] = ((w - 1) * smoothed[t-1] + x[t]) / w   (t > w)
//
// Returns NaN during the seed-collection phase (first window_size - 1
// values), then the SMA seed at sample w, then the recursive form.

#include <limits>
#include <stdexcept>

namespace screamer::detail {

class WilderSmoother {
public:
    explicit WilderSmoother(int window_size) : window_size_(window_size) {
        if (window_size < 1) {
            throw std::invalid_argument("Window size must be positive.");
        }
    }

    void reset() {
        seed_sum_ = 0.0;
        smoothed_ = 0.0;
        n_seen_ = 0;
    }

    double append(double x) {
        n_seen_++;
        if (n_seen_ < window_size_) {
            seed_sum_ += x;
            return std::numeric_limits<double>::quiet_NaN();
        }
        if (n_seen_ == window_size_) {
            seed_sum_ += x;
            smoothed_ = seed_sum_ / window_size_;
        } else {
            const double w = static_cast<double>(window_size_);
            smoothed_ = ((w - 1.0) * smoothed_ + x) / w;
        }
        return smoothed_;
    }

private:
    const int window_size_;
    double seed_sum_ = 0.0;
    double smoothed_ = 0.0;
    int n_seen_ = 0;
};

}  // namespace screamer::detail

#endif
