#ifndef SCREAMER_VWAP_H
#define SCREAMER_VWAP_H

// RollingVWAP: rolling volume-weighted average price using the
// typical price as the weighting basis.
//
//     TP[t]   = (high + low + close) / 3
//     VWAP[t] = sum(TP * volume, w) / sum(volume, w)
//
// 4 -> 1 over (high, low, close, volume). Matches the convention
// pandas-ta-classic uses for vwap(). For a session-VWAP (cumulative
// since some reset point) call reset() at the session boundary --
// the window-bounded version is what we expose by default.

#include <limits>
#include <stdexcept>
#include "screamer/common/functor_base.h"
#include "screamer/detail/rolling_sum.h"

namespace screamer {

class RollingVWAP : public FunctorBase<RollingVWAP, 4, 1> {
public:
    explicit RollingVWAP(int window_size)
        : window_size_(window_size),
          sum_pv_(window_size, "expanding"),
          sum_v_(window_size, "expanding")
    {
        if (window_size < 1) {
            throw std::invalid_argument("Window size must be positive.");
        }
    }

    void reset() override {
        sum_pv_.reset();
        sum_v_.reset();
        n_seen_ = 0;
    }

    ResultTuple call(const InputArray& inputs) override {
        const double high   = inputs[0];
        const double low    = inputs[1];
        const double close  = inputs[2];
        const double volume = inputs[3];
        const double tp = (high + low + close) / 3.0;
        const double pv = sum_pv_.append(tp * volume);
        const double sv = sum_v_.append(volume);
        if (n_seen_ < window_size_) {
            n_seen_++;
            if (n_seen_ < window_size_) {
                return std::numeric_limits<double>::quiet_NaN();
            }
        }
        if (sv == 0.0) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        return pv / sv;
    }

private:
    const int window_size_;
    screamer::detail::RollingSum sum_pv_;
    screamer::detail::RollingSum sum_v_;
    int n_seen_ = 0;
};

}  // namespace screamer

#endif
