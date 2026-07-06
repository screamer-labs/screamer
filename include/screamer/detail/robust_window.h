#ifndef SCREAMER_DETAIL_ROBUST_WINDOW_H
#define SCREAMER_DETAIL_ROBUST_WINDOW_H

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace screamer {
namespace detail {

// A trailing window of the most recent `window_size` pushed values that can
// report the window median and the median absolute deviation (MAD) about that
// median. Both are O(W) per query via std::nth_element. The median follows the
// numpy convention: the average of the two central order statistics for an even
// count. Used by the robust despiking functors (RollingMedianAD, Hampel,
// ImpulseClip) so a single, tested implementation backs all of them.
class RobustWindow {
public:
    explicit RobustWindow(int window_size)
        : window_size_(window_size),
          buffer_(window_size, 0.0),
          scratch_(window_size, 0.0)
    {
        if (window_size <= 0) {
            throw std::invalid_argument("Window size must be positive.");
        }
        reset();
    }

    // Empty window (strict / expanding policies start here).
    void reset() {
        std::fill(buffer_.begin(), buffer_.end(), 0.0);
        index_ = 0;
        size_ = 0;
        last_ = 0;
    }

    // Treat the window as already full of zeros (the "zero" start policy:
    // missing samples count as 0). Call after reset().
    void prefill_zero() {
        size_ = window_size_;
    }

    void push(double value) {
        last_ = index_;
        buffer_[index_] = value;
        if (++index_ == window_size_) {
            index_ = 0;
        }
        if (size_ < window_size_) {
            ++size_;
        }
    }

    // Overwrite the most recently pushed value. Used to keep a detected outlier
    // out of the medians computed for future samples.
    void replace_last(double value) {
        buffer_[last_] = value;
    }

    int count() const { return size_; }
    bool full() const { return size_ == window_size_; }

    double median() {
        load_scratch();
        return median_of(scratch_, size_);
    }

    double mad() {
        double med, mad_value;
        median_and_mad(med, mad_value);
        return mad_value;
    }

    // Compute the median and the MAD together (one scratch load).
    void median_and_mad(double& median_out, double& mad_out) {
        load_scratch();
        median_out = median_of(scratch_, size_);
        for (int i = 0; i < size_; ++i) {
            scratch_[i] = std::abs(scratch_[i] - median_out);
        }
        mad_out = median_of(scratch_, size_);
    }

private:
    // Copy the `size_` active values into scratch_. During warmup the active
    // values occupy indices [0, size_); once full, all `window_size_` slots are
    // active. Order is irrelevant for a median.
    void load_scratch() {
        for (int i = 0; i < size_; ++i) {
            scratch_[i] = buffer_[i];
        }
    }

    // Median of the first n entries of v (partially reorders v). n > 0.
    static double median_of(std::vector<double>& v, int n) {
        const int k = n / 2;
        std::nth_element(v.begin(), v.begin() + k, v.begin() + n);
        const double hi = v[k];
        if (n & 1) {
            return hi;
        }
        const double lo = *std::max_element(v.begin(), v.begin() + k);
        return 0.5 * (lo + hi);
    }

    const int window_size_;
    int index_ = 0;
    int size_ = 0;
    int last_ = 0;
    std::vector<double> buffer_;
    std::vector<double> scratch_;
};

}  // namespace detail
}  // namespace screamer

#endif  // SCREAMER_DETAIL_ROBUST_WINDOW_H
