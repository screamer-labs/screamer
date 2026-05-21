#ifndef SCREAMER_ADX_H
#define SCREAMER_ADX_H

// ADX: Average Directional Index (Wilder, 1978). Measures trend
// strength (not direction). Returns (+DI, -DI, ADX) per step.
//
//     TR[t]   = max(H - L, |H - prev_C|, |L - prev_C|)
//     +DM[t]  = max(H - prev_H, 0) if H - prev_H > prev_L - L else 0
//     -DM[t]  = max(prev_L - L, 0) if prev_L - L > H - prev_H else 0
//
// TR / +DM / -DM are smoothed using TA-Lib's "ADX-flavour" Wilder:
// accumulate (timeperiod - 1) values during warmup, then apply the
// sum-form recurrence
//
//     prev = prev * (1 - 1/w) + new
//
// starting at index `window_size`. The +DI / -DI ratios cancel the
// 1/w scaling so the seed produces output identical to TA-Lib.
//
// ADX is Wilder-smoothed DX in the "average-form": SMA seed over the
// first `window_size` DX values (indices `window_size`..`2*window-1`),
// then standard ((w-1)*prev + new)/w recurrence.
//
// First valid +DI / -DI at sample index `window_size`. First valid
// ADX at sample index `2*window_size - 1`. Bit-exact to TA-Lib.

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <tuple>
#include "screamer/common/float_info.h"
#include "screamer/common/functor_base.h"

namespace screamer {

class ADX : public FunctorBase<ADX, 3, 3> {
public:
    explicit ADX(int window_size = 14) : w_(window_size) {
        if (window_size < 2) {
            throw std::invalid_argument("Window size must be at least 2.");
        }
    }

    void reset() override {
        sum_tr_ = sum_pdm_ = sum_mdm_ = 0.0;
        prev_tr_ = prev_pdm_ = prev_mdm_ = 0.0;
        sum_dx_ = 0.0;
        prev_adx_ = 0.0;
        prev_high_ = std::numeric_limits<double>::quiet_NaN();
        prev_low_  = std::numeric_limits<double>::quiet_NaN();
        prev_close_ = std::numeric_limits<double>::quiet_NaN();
        n_seen_ = 0;
    }

    ResultTuple call(const InputArray& inputs) override {
        const double high  = inputs[0];
        const double low   = inputs[1];
        const double close = inputs[2];
        const double nan = std::numeric_limits<double>::quiet_NaN();

        if (isnan2(high) || isnan2(low) || isnan2(close)) {
            // NaN policy "ignore": leave running state alone, emit NaN.
            return std::make_tuple(nan, nan, nan);
        }

        if (isnan2(prev_close_)) {
            // t = 0: no previous bar; nothing to compute.
            prev_high_ = high;
            prev_low_  = low;
            prev_close_ = close;
            return std::make_tuple(nan, nan, nan);
        }

        // Per-bar TR, +DM, -DM.
        const double tr = std::max({
            high - low,
            std::abs(high - prev_close_),
            std::abs(low  - prev_close_),
        });
        const double up   = high - prev_high_;
        const double down = prev_low_ - low;
        const double pdm = (up > down && up > 0.0) ? up : 0.0;
        const double mdm = (down > up && down > 0.0) ? down : 0.0;

        prev_high_ = high;
        prev_low_  = low;
        prev_close_ = close;

        n_seen_++;
        // Sample-index timeline (0-indexed):
        //   1 .. w-1  : accumulate sums, output NaN
        //   w         : apply first recurrence, emit +DI / -DI; ADX still NaN
        //   w+1..2w-1 : keep recurring; accumulate DX into sum_dx_
        //   2w-1      : emit first ADX = sum_dx_ / w
        //   > 2w-1    : continue with avg-form Wilder on DX
        if (n_seen_ < w_) {
            sum_tr_  += tr;
            sum_pdm_ += pdm;
            sum_mdm_ += mdm;
            return std::make_tuple(nan, nan, nan);
        }

        // Recurrence: prev = prev * (1 - 1/w) + new   (TA-Lib's sum form).
        const double wd = static_cast<double>(w_);
        if (n_seen_ == w_) {
            // First time entering the recurrence; carry over accumulated sums.
            prev_tr_  = sum_tr_  * (wd - 1.0) / wd + tr;
            prev_pdm_ = sum_pdm_ * (wd - 1.0) / wd + pdm;
            prev_mdm_ = sum_mdm_ * (wd - 1.0) / wd + mdm;
        } else {
            prev_tr_  = prev_tr_  * (wd - 1.0) / wd + tr;
            prev_pdm_ = prev_pdm_ * (wd - 1.0) / wd + pdm;
            prev_mdm_ = prev_mdm_ * (wd - 1.0) / wd + mdm;
        }

        if (prev_tr_ <= 0.0) {
            return std::make_tuple(nan, nan, nan);
        }
        const double plus_di  = 100.0 * prev_pdm_ / prev_tr_;
        const double minus_di = 100.0 * prev_mdm_ / prev_tr_;
        const double sum_di = plus_di + minus_di;
        const double dx = (sum_di > 0.0)
                        ? 100.0 * std::abs(plus_di - minus_di) / sum_di
                        : 0.0;

        // Build the ADX SMA seed from DX values at indices w..2w-1
        // (that is `w` DX values total).
        if (n_seen_ < 2 * w_ - 1) {
            sum_dx_ += dx;
            return std::make_tuple(plus_di, minus_di, nan);
        }
        if (n_seen_ == 2 * w_ - 1) {
            sum_dx_ += dx;
            prev_adx_ = sum_dx_ / wd;
        } else {
            // Standard average-form Wilder recurrence on DX.
            prev_adx_ = ((wd - 1.0) * prev_adx_ + dx) / wd;
        }
        return std::make_tuple(plus_di, minus_di, prev_adx_);
    }

private:
    const int w_;
    double sum_tr_ = 0.0;
    double sum_pdm_ = 0.0;
    double sum_mdm_ = 0.0;
    double prev_tr_ = 0.0;
    double prev_pdm_ = 0.0;
    double prev_mdm_ = 0.0;
    double sum_dx_ = 0.0;
    double prev_adx_ = 0.0;
    double prev_high_ = std::numeric_limits<double>::quiet_NaN();
    double prev_low_  = std::numeric_limits<double>::quiet_NaN();
    double prev_close_ = std::numeric_limits<double>::quiet_NaN();
    int n_seen_ = 0;
};

}  // namespace screamer

#endif
