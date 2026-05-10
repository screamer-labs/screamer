#ifndef SCREAMER_ROLLING_MAD_H
#define SCREAMER_ROLLING_MAD_H

// RollingMad: rolling mean absolute deviation,
//
//     MAD[t] = (1 / n) * sum_{i in window} |x_i - mean[t]|
//
// The mean shifts every step, so a closed-form O(1) update is
// impossible: every shift in the mean propagates through all W
// absolute deviations, which (unlike (x_i - mean)^2 in variance) do
// not collapse algebraically. Best practical algorithm is O(W) per
// step, the same approach pandas uses.
//
// Implementation: maintain a circular buffer of the window's values
// plus a running sum to get the mean in O(1). Each step then loops
// over the window once to accumulate sum |x_i - mean|. Validated in
// tests against the manual numpy reference and pandas via a custom
// .apply(np.mean(np.abs(...))).

#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
#include "screamer/common/base.h"
#include "screamer/detail/start_policy.h"

namespace screamer {

class RollingMad : public ScreamerBase {
public:
    RollingMad(int window_size, const std::string& start_policy = "strict")
        : window_size_(window_size),
          start_policy_(detail::parse_start_policy(start_policy))
    {
        if (window_size <= 0) {
            throw std::invalid_argument("Window size must be positive.");
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

private:
    double process_scalar(double newValue) override {
        // Rolling-sum recurrence: same primitive RollingMean uses.
        const double oldValue = buffer_[index_];
        buffer_[index_] = newValue;
        index_++;
        if (index_ == window_size_) {
            index_ = 0;
        }

        if (size_ == window_size_) {
            sum_ += newValue - oldValue;
        } else {
            sum_ += newValue;
            size_++;
        }

        // Determine whether to emit a value and what window size to use.
        const int n_active = size_;  // samples actually held in the buffer
        if (n_active < window_size_) {
            if (start_policy_ == detail::StartPolicy::Strict) {
                return std::numeric_limits<double>::quiet_NaN();
            }
            // Expanding / Zero: still compute MAD with what we have.
        }

        const double divisor = (start_policy_ == detail::StartPolicy::Zero)
                                   ? static_cast<double>(window_size_)
                                   : static_cast<double>(n_active);
        const double mean = sum_ / divisor;

        double abs_sum = 0.0;
        for (int i = 0; i < n_active; ++i) {
            abs_sum += std::abs(buffer_[i] - mean);
        }
        // Zero-policy treats missing samples as zero; their contribution
        // to abs_sum is |0 - mean| = |mean| * (window_size - n_active).
        if (start_policy_ == detail::StartPolicy::Zero && n_active < window_size_) {
            abs_sum += std::abs(mean) * (window_size_ - n_active);
        }
        return abs_sum / divisor;
    }

    const int window_size_;
    const detail::StartPolicy start_policy_;
    std::vector<double> buffer_;
    int index_ = 0;
    int size_ = 0;
    double sum_ = 0.0;
};

}  // namespace screamer

#endif
