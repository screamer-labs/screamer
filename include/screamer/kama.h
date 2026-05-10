#ifndef SCREAMER_KAMA_H
#define SCREAMER_KAMA_H

// KAMA: Kaufman's Adaptive Moving Average (Perry Kaufman, 1998).
//
// Smooths the input with a smoothing-constant SC that adapts to the
// "efficiency" of the recent price action. When the input moves
// monotonically (most of its absolute travel is net displacement),
// ER is close to 1 and SC is close to the fast EMA's; when the input
// is noisy and round-trips (much travel, little net displacement),
// ER is close to 0 and SC is close to the slow EMA's.
//
//   direction[t]  = | x[t] - x[t-n] |
//   volatility[t] = sum_{i=1..n} | x[t-i+1] - x[t-i] |
//   ER[t]         = direction / volatility       (in [0, 1])
//   fast_alpha    = 2 / (fast + 1)
//   slow_alpha    = 2 / (slow + 1)
//   SC[t]         = (ER * (fast_alpha - slow_alpha) + slow_alpha)^2
//   KAMA[t]       = KAMA[t-1] + SC * (x[t] - KAMA[t-1])
//
// The recurrence is seeded at the first valid sample (index n) with
// KAMA[n-1] = x[n-1], matching TA-Lib. First valid output is at t=n.
//
// Composition: one detail::DelayBuffer(n) for x[t-n], one
// detail::RollingSum(n) for the volatility, one prev_x_ scalar for
// the one-step delta, and one running KAMA scalar. All O(1) per step.

#include <cmath>
#include <limits>
#include <stdexcept>
#include "screamer/common/base.h"
#include "screamer/common/float_info.h"
#include "screamer/detail/delay_buffer.h"
#include "screamer/detail/rolling_sum.h"

namespace screamer {

class KAMA : public ScreamerBase {
public:
    KAMA(int window_size, int fast = 2, int slow = 30)
        : window_size_(window_size),
          fast_(fast),
          slow_(slow),
          fast_alpha_(2.0 / (fast + 1.0)),
          slow_alpha_(2.0 / (slow + 1.0)),
          alpha_diff_(fast_alpha_ - slow_alpha_),
          lag_buffer_(window_size, "strict"),
          rolling_abs_diff_(window_size, "expanding")
    {
        if (window_size < 2) {
            throw std::invalid_argument("Window size must be 2 or more.");
        }
        if (fast < 1 || slow < 1) {
            throw std::invalid_argument("fast and slow must be positive.");
        }
        reset();
    }

    void reset() override {
        lag_buffer_.reset();
        rolling_abs_diff_.reset();
        prev_x_ = std::numeric_limits<double>::quiet_NaN();
        kama_ = 0.0;
        n_seen_ = 0;
    }

    double process_scalar(double x) override {
        // 1. One-step abs delta, push into volatility rolling sum.
        const double x_prev = prev_x_;
        const double abs_dx = isnan2(x_prev) ? 0.0 : std::abs(x - x_prev);
        prev_x_ = x;
        const double volatility = rolling_abs_diff_.append(abs_dx);

        // 2. Lagged value x[t-n]. Always advance the buffer; the
        //    return value is meaningful only once we have n+1 samples.
        const double x_lagged = lag_buffer_.append(x);

        n_seen_++;
        if (n_seen_ <= window_size_) {
            return std::numeric_limits<double>::quiet_NaN();
        }

        // 3. Efficiency ratio (in [0, 1] by triangle inequality).
        const double direction = std::abs(x - x_lagged);
        const double er = (volatility > 0.0) ? direction / volatility : 0.0;

        // 4. Smoothing constant.
        const double sc_root = er * alpha_diff_ + slow_alpha_;
        const double sc = sc_root * sc_root;

        // 5. KAMA recurrence. First valid output seeds the previous
        //    KAMA with x[t-1] (= x_prev captured before overwrite).
        if (n_seen_ == window_size_ + 1) {
            kama_ = x_prev;
        }
        kama_ += sc * (x - kama_);
        return kama_;
    }

private:
    const int window_size_;
    const int fast_;
    const int slow_;
    const double fast_alpha_;
    const double slow_alpha_;
    const double alpha_diff_;
    detail::DelayBuffer lag_buffer_;
    detail::RollingSum rolling_abs_diff_;
    double prev_x_ = std::numeric_limits<double>::quiet_NaN();
    double kama_ = 0.0;
    int n_seen_ = 0;
};

}  // namespace screamer

#endif
