#ifndef SCREAMER_IMPULSE_CLIP_H
#define SCREAMER_IMPULSE_CLIP_H

// ImpulseClip: a causal impulse / glitch remover for non-stationary signals.
//
// A spike is a large, isolated jump. In a slowly varying level a spike is hard
// to separate from the signal's own swing, but in the sample-to-sample change it
// stands out: the smooth signal barely moves between neighbours while a spike
// leaps. So detection runs on the first difference, whose scale is trend-free:
//
//     jump   = x[t] - x[t-1]
//     scale  = 1.4826 * MedianAD( jump over the trailing window )
//     outlier if | jump | > n_sigma * scale
//
// A flagged sample is replaced by the trailing median of the values.
//
// Strictly causal (zero latency), so batch == stream exactly. Because an impulse
// is a +/- doublet in the difference (a jump onto the spike and a jump back),
// zero-latency detection flags both the spike and its return sample, so each
// spike replaces two consecutive samples (the second nudged to the median), and
// a genuine step's onset loses one sample. Single-replacement / step-preservation
// would need a 1-sample lookahead, which is deliberately not done.
//
// output modes:
//   "cleaned" : cleaned signal (outliers replaced by the median)  [default]
//   "flag"    : outlier flag (1.0 where an outlier is detected, else 0.0)
//   "nan"     : input with outliers replaced by NaN

#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

#include "screamer/common/base.h"
#include "screamer/common/float_info.h"
#include "screamer/detail/robust_window.h"
#include "screamer/detail/start_policy.h"

namespace screamer {

class ImpulseClip : public ScreamerBase {
public:
    static int parse_output(const std::string& s) {
        if (s == "cleaned") return 0;
        if (s == "flag")    return 1;
        if (s == "nan")     return 2;
        throw std::invalid_argument(
            "output must be \"cleaned\", \"flag\", or \"nan\".");
    }

    ImpulseClip(
        int window_size,
        double n_sigma = 4.0,
        const std::string& output = "cleaned",
        const std::string& start_policy = "strict"
    )
        : n_sigma_(n_sigma),
          output_(parse_output(output)),
          start_policy_(detail::parse_start_policy(start_policy)),
          value_window_(window_size),
          diff_window_(window_size)
    {
        if (window_size <= 0) {
            throw std::invalid_argument("Window size must be positive.");
        }
        if (n_sigma_ <= 0.0) {
            throw std::invalid_argument("n_sigma must be positive.");
        }
        reset();
    }

    void reset() override {
        value_window_.reset();
        diff_window_.reset();
        if (start_policy_ == detail::StartPolicy::Zero) {
            // Missing samples count as 0, so the previous value starts at 0 and
            // both windows start full of zeros.
            value_window_.prefill_zero();
            diff_window_.prefill_zero();
            prev_ = 0.0;
            have_prev_ = true;
        } else {
            prev_ = std::numeric_limits<double>::quiet_NaN();
            have_prev_ = false;
        }
    }

private:
    static constexpr double kMadToStd = 1.4826;

    double process_scalar(double newValue) override {
        // NaN policy "ignore": leave state untouched and emit NaN. prev_ is kept,
        // so the next finite sample differences across the gap.
        if (isnan2(newValue)) {
            return std::numeric_limits<double>::quiet_NaN();
        }

        // The very first finite sample (strict / expanding) has no predecessor,
        // so there is no jump to test yet.
        if (!have_prev_) {
            value_window_.push(newValue);
            prev_ = newValue;
            have_prev_ = true;
            return std::numeric_limits<double>::quiet_NaN();
        }

        const double jump = newValue - prev_;
        prev_ = newValue;  // always the raw previous value
        value_window_.push(newValue);
        diff_window_.push(jump);

        if (start_policy_ == detail::StartPolicy::Strict && !diff_window_.full()) {
            return std::numeric_limits<double>::quiet_NaN();
        }

        const double scale = kMadToStd * diff_window_.mad();
        const bool is_outlier =
            (scale > 0.0) && (std::abs(jump) > n_sigma_ * scale);

        if (is_outlier) {
            const double median = value_window_.median();
            value_window_.replace_last(median);  // keep the window clean
            if (output_ == 1) return 1.0;
            if (output_ == 2) return std::numeric_limits<double>::quiet_NaN();
            return median;
        }

        if (output_ == 1) return 0.0;
        return newValue;  // output 0 and 2 pass a clean sample through
    }

    const double n_sigma_;
    const int output_;
    const detail::StartPolicy start_policy_;
    detail::RobustWindow value_window_;
    detail::RobustWindow diff_window_;
    double prev_ = std::numeric_limits<double>::quiet_NaN();
    bool have_prev_ = false;
};

}  // namespace screamer

#endif  // SCREAMER_IMPULSE_CLIP_H
