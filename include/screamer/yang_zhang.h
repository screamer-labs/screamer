#ifndef SCREAMER_YANG_ZHANG_H
#define SCREAMER_YANG_ZHANG_H

// Yang-Zhang volatility (Yang & Zhang, 2000). The most efficient
// classical range-based estimator, combining three components:
//
//     sigma2_o   = sample variance of overnight log returns
//                  ln(O[t] / C[t-1]) over the window
//     sigma2_c   = sample variance of open-to-close log returns
//                  ln(C[t] / O[t]) over the window
//     sigma2_rs  = mean of per-bar Rogers-Satchell estimates
//     k          = 0.34 / (1.34 + (n+1)/(n-1))
//     sigma2_YZ  = sigma2_o + k * sigma2_c + (1-k) * sigma2_rs
//
// 4 -> 1 over (open, high, low, close). Drift-robust (via the RS
// component) AND handles overnight gaps (via the overnight component).
// ~14x efficiency vs close-to-close volatility for the same n.
//
// First valid output at sample index `window_size` -- we need n+1
// price samples to form n overnight returns.
//
// Composition: holds two RollingVar (overnight + open-to-close log
// returns) and one detail::RollingMean (Rogers-Satchell per-bar).
// All inner smoothers run with start_policy="expanding" so they
// never poison their own state with NaN; the YZ class gates the
// final output itself.

#include <cmath>
#include <limits>
#include <stdexcept>
#include "screamer/common/float_info.h"
#include "screamer/common/functor_base.h"
#include "screamer/detail/rolling_mean.h"
#include "screamer/rolling_var.h"

namespace screamer {

class RollingYangZhangVar : public FunctorBase<RollingYangZhangVar, 4, 1> {
public:
    explicit RollingYangZhangVar(int window_size)
        : window_size_(window_size),
          k_(0.34 / (1.34 + double(window_size + 1) / double(window_size - 1))),
          var_overnight_(window_size, "expanding"),
          var_open_close_(window_size, "expanding"),
          rs_mean_(static_cast<size_t>(window_size), "expanding")
    {
        if (window_size < 2) {
            throw std::invalid_argument("Window size must be at least 2.");
        }
    }

    void reset() override {
        var_overnight_.reset();
        var_open_close_.reset();
        rs_mean_.reset();
        prev_close_ = std::numeric_limits<double>::quiet_NaN();
        n_seen_ = 0;
    }

    ResultTuple call(const InputArray& inputs) override {
        const double O = inputs[0];
        const double H = inputs[1];
        const double L = inputs[2];
        const double C = inputs[3];

        // Open-to-close log return; RS per-bar.
        const double oc = std::log(C / O);
        const double rs = std::log(H / C) * std::log(H / O)
                        + std::log(L / C) * std::log(L / O);

        // Feed the smoothers that take per-bar quantities defined from t=0.
        const double v_oc = var_open_close_.process_scalar(oc);
        const double v_rs = rs_mean_.append(rs);

        // Overnight log return is undefined at t=0 (no prev close).
        double v_on = std::numeric_limits<double>::quiet_NaN();
        if (!isnan2(prev_close_)) {
            const double on = std::log(O / prev_close_);
            v_on = var_overnight_.process_scalar(on);
        }
        prev_close_ = C;

        n_seen_++;
        // First valid YZ output is at sample index n (= window_size),
        // i.e. after we've processed window_size + 1 price bars (so we
        // have window_size overnight returns and window_size OC / RS
        // values).
        if (n_seen_ <= window_size_) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        return v_on + k_ * v_oc + (1.0 - k_) * v_rs;
    }

private:
    const int window_size_;
    const double k_;
    RollingVar var_overnight_;
    RollingVar var_open_close_;
    screamer::detail::RollingMean rs_mean_;
    double prev_close_ = std::numeric_limits<double>::quiet_NaN();
    int n_seen_ = 0;
};


class RollingYangZhangVol : public FunctorBase<RollingYangZhangVol, 4, 1> {
public:
    explicit RollingYangZhangVol(int window_size) : var_(window_size) {}
    void reset() override { var_.reset(); }
    ResultTuple call(const InputArray& inputs) override {
        return std::sqrt(var_.call(inputs));
    }
private:
    RollingYangZhangVar var_;
};

}  // namespace screamer

#endif
