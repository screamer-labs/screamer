#ifndef SCREAMER_ROLLING_BIPOWER_H
#define SCREAMER_ROLLING_BIPOWER_H

// RollingBipowerVar and RollingJumpShare (Barndorff-Nielsen and Shephard
// 2004). On a window of n returns (n = window_size once full),
//     BV = (pi / 2) * sum_{i=2..n} |r_i| |r_{i-1}| * n / (n - 1)
//     RV = sum_{i=1..n} r_i^2
// BV estimates the variance of the continuous part of the returns and RV the
// whole variance, jumps included; the factor n / (n - 1) scales the n - 1
// pairs to the n terms of RV. RollingJumpShare = max(0, 1 - BV / RV), the
// share of RV that comes from jumps, and 0 when RV is 0.
//
// start_policy: "strict" gives NaN until the window holds window_size
// returns, "expanding" uses the returns seen so far (two at least), "zero"
// starts from a window of zeros. nan_policy: ignore (a NaN is not stored;
// the pair after it joins the returns on either side). O(1) per step.

#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include "screamer/common/base.h"
#include "screamer/common/float_info.h"
#include "screamer/detail/rolling_sum.h"
#include "screamer/detail/start_policy.h"

namespace screamer {
namespace detail {

class BipowerWindow {
public:
    BipowerWindow(int window_size, const std::string& start_policy)
        : window_size_(window_size),
          policy_(parse_start_policy(start_policy)),
          pairs_(window_size >= 2 ? static_cast<std::size_t>(window_size - 1) : 1, start_policy),
          squares_(window_size >= 2 ? static_cast<std::size_t>(window_size) : 2, start_policy)
    {
        if (window_size < 2) throw std::invalid_argument("Window size must be 2 or more.");
        reset();
    }

    void reset() {
        pairs_.reset();
        squares_.reset();
        const bool zero = policy_ == StartPolicy::Zero;
        n_ = zero ? static_cast<std::size_t>(window_size_) : 0;
        has_prev_ = zero;               // the zero window's last return is 0
        prev_abs_ = 0.0;
    }

    // Adds a finite return; bv and rv are NaN where the policy gives none.
    void push(double r, double& bv, double& rv) {
        constexpr double kHalfPi = 1.5707963267948966;
        const double nan = std::numeric_limits<double>::quiet_NaN();
        if (n_ < static_cast<std::size_t>(window_size_)) ++n_;
        rv = squares_.append(r * r);
        const double a = std::fabs(r);
        const double pair_sum = has_prev_ ? pairs_.append(prev_abs_ * a) : nan;
        prev_abs_ = a;
        has_prev_ = true;
        if (n_ < 2) {
            bv = nan;
            return;
        }
        const double n = static_cast<double>(n_);
        const double scale = kHalfPi * n / (n - 1.0);
        bv = pair_sum * scale;
    }

private:
    const int window_size_;
    const StartPolicy policy_;
    RollingSum pairs_, squares_;
    std::size_t n_ = 0;
    bool has_prev_ = false;
    double prev_abs_ = 0.0;
};

}  // namespace detail

class RollingBipowerVar : public ScreamerBase {
public:
    RollingBipowerVar(int window_size = 20, const std::string& start_policy = "strict")
        : window_(window_size, start_policy) {}

    void reset() override { window_.reset(); }

    double process_scalar(double x) override {
        if (isnan2(x)) return std::numeric_limits<double>::quiet_NaN();
        double bv = 0.0, rv = 0.0;
        window_.push(x, bv, rv);
        return bv;
    }

private:
    detail::BipowerWindow window_;
};

class RollingJumpShare : public ScreamerBase {
public:
    RollingJumpShare(int window_size = 20, const std::string& start_policy = "strict")
        : window_(window_size, start_policy) {}

    void reset() override { window_.reset(); }

    double process_scalar(double x) override {
        const double nan = std::numeric_limits<double>::quiet_NaN();
        if (isnan2(x)) return nan;
        double bv = 0.0, rv = 0.0;
        window_.push(x, bv, rv);
        if (isnan2(bv) || isnan2(rv)) return nan;
        if (rv == 0.0) return 0.0;
        const double share = 1.0 - bv / rv;
        return share > 0.0 ? share : 0.0;
    }

private:
    detail::BipowerWindow window_;
};

}  // namespace screamer

#endif  // SCREAMER_ROLLING_BIPOWER_H
