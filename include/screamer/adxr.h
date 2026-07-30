#ifndef SCREAMER_ADXR_H
#define SCREAMER_ADXR_H

// ADXR: Wilder's average directional index rating, the mean of ADX now and
// ADX (window_size - 1) bars ago. Smooths ADX to gauge trend-strength
// momentum. Shares DmiCore with ADX.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>
#include "screamer/common/float_info.h"
#include "screamer/common/functor_base.h"
#include "screamer/detail/dmi_core.h"

namespace screamer {

class ADXR : public FunctorBase<ADXR, 3, 1> {
public:
    explicit ADXR(int window_size = 14)
        : core_(window_size),
          lag_(window_size - 1),
          buf_(static_cast<size_t>(window_size),
               std::numeric_limits<double>::quiet_NaN()) {}

    void reset() override {
        core_.reset();
        std::fill(buf_.begin(), buf_.end(),
                   std::numeric_limits<double>::quiet_NaN());
        pos_ = 0;
    }

    ResultTuple call(const InputArray& inputs) override {
        // nan_policy: ignore. A bar with any missing field is skipped whole.
        // Without this the lag buffer took the NaN and pos_ advanced, so the
        // bar consumed a slot and the ADX read back `window_size - 1` bars
        // later was the wrong one: a single NaN shifted every later output.
        if (any_nan(inputs[0], inputs[1], inputs[2])) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        const double adx = core_.update(inputs[0], inputs[1], inputs[2]).adx;
        // buf_ holds the last window_size adx values (including NaN, by bar).
        // The slot `lag_` bars behind the write cursor holds ADX[t - lag_].
        const size_t back = static_cast<size_t>(lag_);
        const size_t idx = (pos_ + buf_.size() - back) % buf_.size();
        const double adx_lag = buf_[idx];
        buf_[pos_] = adx;
        pos_ = (pos_ + 1) % buf_.size();
        if (std::isnan(adx) || std::isnan(adx_lag)) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        return (adx + adx_lag) / 2.0;
    }

private:
    detail::DmiCore core_;
    int lag_;
    std::vector<double> buf_;
    size_t pos_ = 0;
};

}  // namespace screamer

#endif  // SCREAMER_ADXR_H
