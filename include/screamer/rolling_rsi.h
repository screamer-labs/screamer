#ifndef SCREAMER_ROLLING_RSI_H
#define SCREAMER_ROLLING_RSI_H

// RollingRSI: Relative Strength Index, J. Welles Wilder Jr. (1978).
//
// Two smoothing methods are supported, picked via the `method`
// constructor argument:
//
//   "wilder" (default)  Wilder's original recursive smoothing
//     avg_gain[t] = ((n-1) * avg_gain[t-1] + gain[t]) / n
//     Seeded at t=n with avg_gain = mean of the first n gains.
//     This is what TA-Lib, pandas-ta, and most charting platforms
//     compute when you ask for "RSI".
//
//   "cutler"            Cutler's RSI -- simple moving average of
//     gains and losses over the period. Equivalent to a naive
//     "smooth gains and losses with rolling().mean()" decomposition.
//     Some quant literature uses this form because it makes the
//     algebra cleaner; charting platforms generally don't.
//
//   RSI[t] = 100 * avg_gain / (avg_gain + avg_loss)
//          = 100 - 100 / (1 + avg_gain / avg_loss)
//
// Both methods are O(1) per step. The `start_policy` argument is
// kept for backward compatibility but applies only to Cutler; it
// has no meaning for Wilder's recursive form (which has its own
// fixed warmup of n samples).
//
// First valid output:
//   wilder: at sample index n (zero-indexed; matches TA-Lib).
//   cutler: at sample index n-1 (one sample earlier; consequence of
//           treating the first sample's missing-delta as a zero
//           contribution to the rolling-sum buffer).
//
// See docs/conventions.md for the rationale and live alignment
// numbers against TA-Lib and pandas-ta-classic.

#include <limits>
#include <stdexcept>
#include <string>
#include "screamer/common/base.h"
#include "screamer/common/float_info.h"
#include "screamer/detail/rolling_sum.h"
#include "screamer/detail/start_policy.h"

namespace screamer {

class RollingRSI : public ScreamerBase {
public:
    enum class Method { Wilder, Cutler };

    RollingRSI(int window_size,
               const std::string& method = "wilder",
               const std::string& start_policy = "strict")
        : window_size_(window_size),
          method_(parse_method(method)),
          start_policy_(detail::parse_start_policy(start_policy)),
          rolling_gain_sum_(window_size, start_policy),
          rolling_loss_sum_(window_size, start_policy)
    {
        if (window_size_ < 2) {
            throw std::invalid_argument("Window size must be 2 or more.");
        }
        reset();
    }

    void reset() override {
        rolling_gain_sum_.reset();
        rolling_loss_sum_.reset();
        prev_x_ = (start_policy_ == detail::StartPolicy::Zero)
                  ? 0.0
                  : std::numeric_limits<double>::quiet_NaN();
        n_ = (start_policy_ != detail::StartPolicy::Zero) ? 0 : window_size_;
        wilder_avg_gain_ = 0.0;
        wilder_avg_loss_ = 0.0;
        wilder_seed_sum_gain_ = 0.0;
        wilder_seed_sum_loss_ = 0.0;
        wilder_count_ = 0;
    }

    double process_scalar(double x) override {
        // NaN policy "ignore": leave Wilder/Cutler state untouched.
        if (isnan2(x)) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        if (method_ == Method::Cutler) {
            return process_cutler(x);
        }
        return process_wilder(x);
    }

private:
    static Method parse_method(const std::string& s) {
        if (s == "wilder") return Method::Wilder;
        if (s == "cutler") return Method::Cutler;
        throw std::invalid_argument("method must be \"wilder\" or \"cutler\"");
    }

    double process_cutler(double x) {
        // Preserves the original behaviour: missing first delta becomes
        // a 0 contribution to the rolling buffer, so first valid output
        // is at sample n-1 (one earlier than TA-Lib). Documented.
        double dx = x - prev_x_;
        prev_x_ = x;

        if (isnan2(dx)) {
            dx = 0.0;
        } else if (n_ < window_size_) {
            n_++;
        }

        const double gain_sum = rolling_gain_sum_.append(dx > 0.0 ? dx : 0.0);
        const double loss_sum = rolling_loss_sum_.append(dx < 0.0 ? -dx : 0.0);
        return 100.0 * gain_sum / (gain_sum + loss_sum);
    }

    double process_wilder(double x) {
        // TA-Lib alignment: skip the first sample entirely (no delta
        // available), then accumulate n real gains/losses, seed
        // avg_gain / avg_loss at sample n, recurrence from sample n+1.
        if (isnan2(prev_x_)) {
            prev_x_ = x;
            return std::numeric_limits<double>::quiet_NaN();
        }
        const double dx = x - prev_x_;
        prev_x_ = x;

        const double gain = dx > 0.0 ? dx : 0.0;
        const double loss = dx < 0.0 ? -dx : 0.0;

        wilder_count_++;
        if (wilder_count_ < window_size_) {
            // Warmup: accumulate the first n-1 gains/losses, no output.
            wilder_seed_sum_gain_ += gain;
            wilder_seed_sum_loss_ += loss;
            return std::numeric_limits<double>::quiet_NaN();
        }
        if (wilder_count_ == window_size_) {
            // First valid sample: seed averages from the first n gains.
            wilder_seed_sum_gain_ += gain;
            wilder_seed_sum_loss_ += loss;
            wilder_avg_gain_ = wilder_seed_sum_gain_ / window_size_;
            wilder_avg_loss_ = wilder_seed_sum_loss_ / window_size_;
        } else {
            // Steady state: Wilder's smoothing recurrence.
            const double w = static_cast<double>(window_size_);
            wilder_avg_gain_ = ((w - 1.0) * wilder_avg_gain_ + gain) / w;
            wilder_avg_loss_ = ((w - 1.0) * wilder_avg_loss_ + loss) / w;
        }

        const double total = wilder_avg_gain_ + wilder_avg_loss_;
        if (total == 0.0) {
            return 50.0;  // flat input: RSI is conventionally 50.
        }
        return 100.0 * wilder_avg_gain_ / total;
    }

    const int window_size_;
    const Method method_;
    const detail::StartPolicy start_policy_;
    detail::RollingSum rolling_gain_sum_;
    detail::RollingSum rolling_loss_sum_;
    double prev_x_ = 0.0;
    int n_ = 0;

    // Wilder state. wilder_count_ counts real (non-first) samples.
    double wilder_avg_gain_ = 0.0;
    double wilder_avg_loss_ = 0.0;
    double wilder_seed_sum_gain_ = 0.0;
    double wilder_seed_sum_loss_ = 0.0;
    int wilder_count_ = 0;
};

}  // namespace screamer

#endif
