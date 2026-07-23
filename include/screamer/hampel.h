#ifndef SCREAMER_HAMPEL_H
#define SCREAMER_HAMPEL_H

// Hampel: the canonical robust despiker (Hampel filter / Hampel identifier),
// causal trailing-window variant.
//
// For each sample, over the trailing window compute the median m and the median
// absolute deviation MAD. A sample is an outlier when
//
//     | x - m | > n_sigma * 1.4826 * MAD
//
// (1.4826 makes MAD a Gaussian-consistent standard-deviation estimate). An
// outlier is replaced by the window median; the replacement, not the raw
// outlier, is fed back into the window so a burst of spikes cannot inflate the
// scale for following samples.
//
// Strictly causal (trailing window only), so batch == stream exactly.
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

class Hampel : public ScreamerBase {
public:
    static int parse_output(const std::string& s) {
        if (s == "cleaned") return 0;
        if (s == "flag")    return 1;
        if (s == "nan")     return 2;
        throw std::invalid_argument(
            "output must be \"cleaned\", \"flag\", or \"nan\".");
    }

    Hampel(
        int window_size,
        double n_sigma = 3.0,
        const std::string& output = "cleaned",
        const std::string& start_policy = "strict"
    )
        : n_sigma_(n_sigma),
          output_(parse_output(output)),
          start_policy_(detail::parse_start_policy(start_policy)),
          window_(window_size)
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
        window_.reset();
        if (start_policy_ == detail::StartPolicy::Zero) {
            window_.prefill_zero();
        }
    }

private:
    static constexpr double kMadToStd = 1.4826;

    double process_scalar(double newValue) override {
        // NaN policy "ignore": leave state untouched and emit NaN.
        if (isnan2(newValue)) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        window_.push(newValue);

        if (start_policy_ == detail::StartPolicy::Strict && !window_.full()) {
            return std::numeric_limits<double>::quiet_NaN();
        }

        double median, mad;
        window_.median_and_mad(median, mad);
        const double scale = kMadToStd * mad;

        const bool is_outlier =
            (scale > 0.0) && (std::abs(newValue - median) > n_sigma_ * scale);

        if (is_outlier) {
            window_.replace_last(median);  // keep the window clean for later
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
    detail::RobustWindow window_;
};

}  // namespace screamer

#endif  // SCREAMER_HAMPEL_H
