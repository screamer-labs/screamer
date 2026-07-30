#ifndef SCREAMER_DETAIL_DMI_CORE_H
#define SCREAMER_DETAIL_DMI_CORE_H

// DmiCore: Wilder directional-movement engine (Wilder, 1978), bit-exact to
// TA-Lib. One update per (high, low, close) bar. All outputs NaN during
// warmup. Shared by ADX and the standalone DMI operators so the smoothing
// recurrence exists in exactly one place. See include/screamer/adx.h history
// for the timeline derivation.

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include "screamer/common/float_info.h"

namespace screamer {
namespace detail {

struct DmiResult {
    double plus_di = std::numeric_limits<double>::quiet_NaN();
    double minus_di = std::numeric_limits<double>::quiet_NaN();
    double plus_dm = std::numeric_limits<double>::quiet_NaN();
    double minus_dm = std::numeric_limits<double>::quiet_NaN();
    double dx = std::numeric_limits<double>::quiet_NaN();
    double adx = std::numeric_limits<double>::quiet_NaN();
};

class DmiCore {
public:
    explicit DmiCore(int window_size) : w_(window_size) {
        if (window_size < 2) {
            throw std::invalid_argument("Window size must be at least 2.");
        }
    }

    void reset() {
        sum_tr_ = sum_pdm_ = sum_mdm_ = 0.0;
        prev_tr_ = prev_pdm_ = prev_mdm_ = 0.0;
        sum_dx_ = 0.0;
        prev_adx_ = 0.0;
        prev_high_ = std::numeric_limits<double>::quiet_NaN();
        prev_low_ = std::numeric_limits<double>::quiet_NaN();
        prev_close_ = std::numeric_limits<double>::quiet_NaN();
        n_seen_ = 0;
    }

    DmiResult update(double high, double low, double close) {
        DmiResult r;
        if (isnan2(high) || isnan2(low) || isnan2(close)) {
            return r;  // NaN policy "ignore": leave state, emit all NaN.
        }
        if (isnan2(prev_close_)) {
            prev_high_ = high;
            prev_low_ = low;
            prev_close_ = close;
            return r;
        }
        const double tr = std::max({high - low, std::abs(high - prev_close_),
                                    std::abs(low - prev_close_)});
        const double up = high - prev_high_;
        const double down = prev_low_ - low;
        const double pdm = (up > down && up > 0.0) ? up : 0.0;
        const double mdm = (down > up && down > 0.0) ? down : 0.0;
        prev_high_ = high;
        prev_low_ = low;
        prev_close_ = close;
        n_seen_++;
        if (n_seen_ < w_) {
            sum_tr_ += tr;
            sum_pdm_ += pdm;
            sum_mdm_ += mdm;
            return r;
        }
        const double wd = static_cast<double>(w_);
        if (n_seen_ == w_) {
            prev_tr_ = sum_tr_ * (wd - 1.0) / wd + tr;
            prev_pdm_ = sum_pdm_ * (wd - 1.0) / wd + pdm;
            prev_mdm_ = sum_mdm_ * (wd - 1.0) / wd + mdm;
        } else {
            prev_tr_ = prev_tr_ * (wd - 1.0) / wd + tr;
            prev_pdm_ = prev_pdm_ * (wd - 1.0) / wd + pdm;
            prev_mdm_ = prev_mdm_ * (wd - 1.0) / wd + mdm;
        }
        r.plus_dm = prev_pdm_;
        r.minus_dm = prev_mdm_;
        if (prev_tr_ <= 0.0) {
            return r;  // DM values valid, DI/DX undefined this bar.
        }
        const double plus_di = 100.0 * prev_pdm_ / prev_tr_;
        const double minus_di = 100.0 * prev_mdm_ / prev_tr_;
        r.plus_di = plus_di;
        r.minus_di = minus_di;
        const double sum_di = plus_di + minus_di;
        const double dx = (sum_di > 0.0)
                              ? 100.0 * std::abs(plus_di - minus_di) / sum_di
                              : 0.0;
        r.dx = dx;
        if (n_seen_ < 2 * w_ - 1) {
            sum_dx_ += dx;
            return r;
        }
        if (n_seen_ == 2 * w_ - 1) {
            sum_dx_ += dx;
            prev_adx_ = sum_dx_ / wd;
        } else {
            prev_adx_ = ((wd - 1.0) * prev_adx_ + dx) / wd;
        }
        r.adx = prev_adx_;
        return r;
    }

    int window() const { return w_; }

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
    double prev_low_ = std::numeric_limits<double>::quiet_NaN();
    double prev_close_ = std::numeric_limits<double>::quiet_NaN();
    int n_seen_ = 0;
};

}  // namespace detail
}  // namespace screamer

#endif  // SCREAMER_DETAIL_DMI_CORE_H
