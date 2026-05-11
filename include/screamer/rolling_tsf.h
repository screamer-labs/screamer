#ifndef SCREAMER_ROLLING_TSF_H
#define SCREAMER_ROLLING_TSF_H

// RollingTSF: Time-Series Forecast (TA-Lib's TSF). Linear regression
// of y vs time index over the trailing window, projected ONE step
// ahead. Matches `talib.TSF(real, timeperiod)` bit-exactly.
//
//     For the window {(0, y_{t-n+1}), (1, y_{t-n+2}), ..., (n-1, y_t)}:
//     fit y = slope * t + intercept,
//     TSF[t] = slope * n + intercept   (the line evaluated at the
//                                       NEXT bar, n steps from window start)
//
// Equivalently: TSF[t] = LINEARREG[t] + LINEARREG_SLOPE[t], using
// TA-Lib's vocabulary.
//
// 1 -> 1. Composes five detail::RollingSum buffers. O(1) per step.
// First valid at sample index window_size - 1.

#include <limits>
#include <stdexcept>
#include "screamer/common/base.h"
#include "screamer/detail/rolling_sum.h"

namespace screamer {

class RollingTSF : public ScreamerBase {
public:
    explicit RollingTSF(int window_size)
        : window_size_(window_size),
          sum_y_(window_size, "expanding"),
          sum_yt_(window_size, "expanding")
    {
        if (window_size < 2) {
            throw std::invalid_argument("Window size must be at least 2.");
        }
        // Precompute time-axis sums for indices 0..n-1.
        const double n = window_size_;
        sum_t_  = n * (n - 1.0) / 2.0;
        sum_tt_ = (n - 1.0) * n * (2.0 * n - 1.0) / 6.0;
        // denom_t = n * sum_tt - sum_t^2 (constant)
        denom_t_ = n * sum_tt_ - sum_t_ * sum_t_;
    }

    void reset() override {
        sum_y_.reset();
        sum_yt_.reset();
        time_idx_ = 0;
        n_seen_ = 0;
    }

    double process_scalar(double y) override {
        // We want a rolling regression of y[t-n+1..t] vs the local
        // time index 0..n-1. To maintain sum(y * t_local) cheaply, we
        // use the identity that updating from one window to the next
        // shifts the time indices down by 1. The combined formula is:
        //     new_sum_yt = old_sum_yt - sum_y_after_remove + (n-1) * y_new
        // but that's awkward to keep track of incrementally with our
        // existing RollingSum primitive, so we recompute via a
        // separate "absolute" sum_y_t_absolute and re-derive at output.
        //
        // Simpler approach: maintain sum_y over the window (1 buffer),
        // and sum(absolute_t * y) over the window. Time indices in the
        // window are time_idx - (n-1), time_idx - (n-2), ..., time_idx.
        // Then sum_local_t * y = sum_absolute_t * y - (time_idx - (n-1)) * sum_y.
        const double t_abs = static_cast<double>(time_idx_);
        const double Sy_total = sum_y_ .append(y);
        const double Sty      = sum_yt_.append(t_abs * y);
        time_idx_++;

        if (n_seen_ < window_size_) {
            n_seen_++;
            if (n_seen_ < window_size_) {
                return std::numeric_limits<double>::quiet_NaN();
            }
        }

        // Convert absolute-time sums to local-time sums over the
        // current window. The window covers absolute time indices
        // [t_abs - (n-1) .. t_abs], so local_t = absolute_t - shift,
        // where shift = t_abs - (n-1).
        const double n = window_size_;
        const double shift = t_abs - (n - 1.0);
        // local sum_t*y = sum_abs_t*y - shift * sum_y.
        const double Sty_local = Sty - shift * Sy_total;

        const double num   = n * Sty_local - sum_t_ * Sy_total;
        const double slope = num / denom_t_;
        const double intercept = (Sy_total - slope * sum_t_) / n;
        // Project one step beyond the window: evaluate at local t = n.
        return slope * n + intercept;
    }

private:
    const int window_size_;
    double sum_t_;
    double sum_tt_;
    double denom_t_;
    detail::RollingSum sum_y_;
    detail::RollingSum sum_yt_;
    long time_idx_ = 0;
    int n_seen_ = 0;
};

}  // namespace screamer

#endif
