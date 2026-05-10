#ifndef SCREAMER_TRIMA_H
#define SCREAMER_TRIMA_H

// TRIMA: Triangular Moving Average -- a double-smoothed simple mean.
// The effective per-sample weights form a symmetric triangle (rising
// then falling), giving more weight to the centre of the window.
//
//     TRIMA[t] = SMA(SMA(x, n_inner), n_outer)[t]
//
// TA-Lib convention for window N:
//     N odd:   n_inner = n_outer = (N + 1) / 2
//     N even:  n_inner = N/2 + 1, n_outer = N/2
// In both cases n_inner + n_outer - 1 = N (the effective triangle width).
//
// Pure composition of two detail::RollingMean instances. Both are run
// with start_policy="expanding" so they always return finite values
// (otherwise the inner's NaN warmup would poison the outer's running
// sum permanently). The TRIMA class itself enforces strict warmup
// (NaN until N samples seen) by tracking a sample counter.
//
// O(1) per step.

#include <limits>
#include <stdexcept>
#include "screamer/common/base.h"
#include "screamer/detail/rolling_mean.h"

namespace screamer {

class TRIMA : public ScreamerBase {
public:
    explicit TRIMA(int window_size)
        : window_size_(window_size),
          n_inner_(compute_inner(window_size)),
          n_outer_(compute_outer(window_size)),
          inner_(static_cast<size_t>(n_inner_), "expanding"),
          outer_(static_cast<size_t>(n_outer_), "expanding")
    {
        if (window_size < 1) {
            throw std::invalid_argument("Window size must be positive.");
        }
    }

    void reset() override {
        inner_.reset();
        outer_.reset();
        n_seen_ = 0;
    }

    double process_scalar(double x) override {
        const double inner_val = inner_.append(x);
        const double outer_val = outer_.append(inner_val);
        if (n_seen_ < window_size_) {
            n_seen_++;
        }
        if (n_seen_ < window_size_) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        return outer_val;
    }

private:
    static int compute_inner(int n) {
        return (n % 2 == 1) ? (n + 1) / 2 : (n / 2 + 1);
    }
    static int compute_outer(int n) {
        return (n % 2 == 1) ? (n + 1) / 2 : (n / 2);
    }

    const int window_size_;
    const int n_inner_;
    const int n_outer_;
    detail::RollingMean inner_;
    detail::RollingMean outer_;
    int n_seen_ = 0;
};

}  // namespace screamer

#endif
