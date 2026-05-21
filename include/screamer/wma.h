#ifndef SCREAMER_WMA_H
#define SCREAMER_WMA_H

// WMA: linearly-weighted moving average. The newest sample in the
// window has weight w, the oldest has weight 1, so
//
//     WMA[t] = (1*x[t-w+1] + 2*x[t-w+2] + ... + w*x[t]) / (w(w+1)/2)
//
// Computed in O(1) per step via the identity
//
//     W[t] - W[t-1] = w*x[t] - S[t-1]
//
// where W[t] is the linear-weighted sum and S[t-1] is the simple
// rolling sum of the *previous* window (i.e. the sum from t-w to
// t-1, before the new sample arrives). The class therefore holds a
// detail::RollingSum (for S) and a single double (for W).
//
// During warmup the recurrence depends on the start policy:
//   strict / expanding: W += n_active * x_new (expanding-form numerator)
//   zero:               W += w * x_new - S[t-1] (zero-padded past)
// Post-warmup, all policies use the rolling recurrence. The two
// numerators agree exactly at the moment the window first fills, so
// the transition is seamless.

#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include "screamer/common/base.h"
#include "screamer/common/float_info.h"
#include "screamer/detail/rolling_sum.h"
#include "screamer/detail/start_policy.h"

namespace screamer {

class WMA : public ScreamerBase {
public:
    WMA(int window_size, const std::string& start_policy = "strict")
        : window_size_(window_size),
          start_policy_(detail::parse_start_policy(start_policy)),
          // Inner RollingSum runs "expanding" so it returns valid sums
          // throughout warmup; this class controls warmup output itself.
          rolling_sum_(window_size, "expanding"),
          weight_sum_full_(window_size * (window_size + 1) / 2.0)
    {
        if (window_size < 1) {
            throw std::invalid_argument("Window size must be positive.");
        }
        reset();
    }

    void reset() override {
        rolling_sum_.reset();
        weighted_sum_ = 0.0;
        prev_simple_sum_ = 0.0;
        n_ = 0;
    }

    // process_scalar is public so other ScreamerBase classes (e.g. HullMA)
    // can chain a WMA inside their own process_scalar without going
    // through the Python dispatcher.
    double process_scalar(double x) override {
        // NaN policy "ignore": leave weighted_sum_, prev_simple_sum_, and n_
        // untouched. See docs/nan_policy.md.
        if (isnan2(x)) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        if (n_ < window_size_) {
            n_++;
            if (start_policy_ == detail::StartPolicy::Zero) {
                // Zero-pad missing past: full rolling recurrence.
                weighted_sum_ += window_size_ * x - prev_simple_sum_;
            } else {
                // Strict/Expanding: expanding-form numerator.
                weighted_sum_ += n_ * x;
            }
        } else {
            // Post-warmup: standard rolling recurrence.
            weighted_sum_ += window_size_ * x - prev_simple_sum_;
        }
        // Always advance the simple-sum buffer; its return value becomes
        // S[t-1] for the next call.
        prev_simple_sum_ = rolling_sum_.append(x);

        if (n_ == window_size_) {
            return weighted_sum_ / weight_sum_full_;
        }
        // Warmup output, by policy.
        switch (start_policy_) {
            case detail::StartPolicy::Strict:
                return std::numeric_limits<double>::quiet_NaN();
            case detail::StartPolicy::Expanding:
                return weighted_sum_ / (n_ * (n_ + 1) / 2.0);
            case detail::StartPolicy::Zero:
                return weighted_sum_ / weight_sum_full_;
        }
        return std::numeric_limits<double>::quiet_NaN();  // unreachable
    }

private:
    const int window_size_;
    const detail::StartPolicy start_policy_;
    detail::RollingSum rolling_sum_;
    const double weight_sum_full_;
    double weighted_sum_ = 0.0;
    double prev_simple_sum_ = 0.0;
    int n_ = 0;
};

}  // namespace screamer

#endif
