#ifndef SCREAMER_HULL_MA_H
#define SCREAMER_HULL_MA_H

// HullMA: Alan Hull's responsive low-lag moving average.
//
//     HullMA[t] = WMA( 2 * WMA(x, n/2) - WMA(x, n),  sqrt(n) )[t]
//
// where the window arguments use integer floor: n/2 = n >> 1,
// sqrt(n) = floor(sqrt(double(n))). The construction subtracts a
// slow WMA from twice a fast WMA (anticipating the trend), then
// smooths the result with a much shorter WMA -- the net effect is
// a smoother that tracks the price closely with less lag than a
// plain SMA / EMA / WMA of comparable window.
//
// Pure composition of three WMA instances. Inner WMAs use
// start_policy="expanding" so they don't emit NaN during warmup
// (which would poison the outer); HullMA enforces strict warmup
// itself by counting samples and emitting NaN until
// n + sqrt(n) - 1 samples have been processed.
//
// O(1) per step.

#include <cmath>
#include <limits>
#include <stdexcept>
#include "screamer/common/base.h"
#include "screamer/wma.h"

namespace screamer {

class HullMA : public ScreamerBase {
public:
    explicit HullMA(int window_size)
        : window_size_(window_size),
          n_half_(window_size / 2),
          n_sqrt_(static_cast<int>(std::sqrt(static_cast<double>(window_size)))),
          full_warmup_(window_size + n_sqrt_ - 1),
          wma_half_(n_half_, "expanding"),
          wma_full_(window_size, "expanding"),
          wma_outer_(n_sqrt_, "expanding")
    {
        if (window_size < 4) {
            // n/2 must be >= 2 (WMA window >= 1), and floor(sqrt(n)) >= 2
            // requires n >= 4. Below that the construction degenerates.
            throw std::invalid_argument("Window size must be at least 4.");
        }
    }

    void reset() override {
        wma_half_.reset();
        wma_full_.reset();
        wma_outer_.reset();
        n_seen_ = 0;
    }

    double process_scalar(double x) override {
        const double w_half = wma_half_.process_scalar(x);
        const double w_full = wma_full_.process_scalar(x);
        const double diff = 2.0 * w_half - w_full;
        const double out = wma_outer_.process_scalar(diff);
        if (n_seen_ < full_warmup_) {
            n_seen_++;
        }
        if (n_seen_ < full_warmup_) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        return out;
    }

private:
    const int window_size_;
    const int n_half_;
    const int n_sqrt_;
    const int full_warmup_;
    WMA wma_half_;
    WMA wma_full_;
    WMA wma_outer_;
    int n_seen_ = 0;
};

}  // namespace screamer

#endif
