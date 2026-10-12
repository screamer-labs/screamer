#ifndef SCREAMER_ROLLING_LINEAR_REGRESSION_H
#define SCREAMER_ROLLING_LINEAR_REGRESSION_H

// RollingLinearRegression: 2 -> 4 OLS fit of (target ~ slope *
// regressor + intercept). Returns the full statistical tuple per step:
//
//     (slope, intercept, r_squared, stderr)
//
// where:
//   slope     = (n*Sxy - Sx*Sy) / (n*Sxx - Sx*Sx)
//   intercept = (Sy - slope*Sx) / n
//   r_squared = (n*Sxy - Sx*Sy)^2 / ((n*Sxx - Sx*Sx) * (n*Syy - Sy*Sy))
//   stderr    = sqrt(SSE / (n - 2)),  SSE = SST - SSR
//
// 2 -> 4 functor over the N->M dispatcher. Convention
// matches RollingBeta: FIRST argument is the dependent (target),
// SECOND is the regressor. The window sums (Sx, Sy, Sxx, Syy, Sxy) are taken
// about an origin near the means (detail::ShiftedWindowCoSums), so the fit
// stays accurate on price levels; every formula above except the intercept is
// unchanged by the shift, and the intercept is mean_y - slope * mean_x.
// O(1) per step.

#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <tuple>
#include "screamer/common/functor_base.h"
#include "screamer/common/float_info.h"
#include "screamer/detail/shifted_window_sums.h"
#include "screamer/detail/start_policy.h"

namespace screamer {

class RollingLinearRegression : public FunctorBase<RollingLinearRegression, 2, 4> {
    using Sums = detail::ShiftedWindowCoSums<true, true>;

public:
    RollingLinearRegression(int window_size, const std::string& start_policy = "strict")
        : start_policy_(detail::parse_start_policy(start_policy)),
          sums_(window_size >= 3 ? window_size : 3, start_policy_)
    {
        if (window_size < 3) {
            throw std::invalid_argument("Window size must be at least 3 (n-2 in stderr).");
        }
        reset();
    }

    void reset() override { sums_.reset(); }

    // One event on the window's state; the batch column loop runs it on a
    // local copy (FunctorBase, has_stepper). Each product is its own statement
    // (see shifted_window_sums.h).
    struct Stepper {
        Sums::Cursor c;
        bool strict;

        inline ResultTuple call(const InputArray& inputs) {
            const double nan = std::numeric_limits<double>::quiet_NaN();
            const double target = inputs[0];  // y (dependent)
            const double regr   = inputs[1];  // x (regressor)
            if (isnan2(target) || isnan2(regr)) {
                return std::make_tuple(nan, nan, nan, nan);
            }
            c.push(regr, target);  // x = regressor, y = target
            const std::size_t n_ = c.st.count;
            if ((strict && n_ < c.cap) || n_ < 3) {
                return std::make_tuple(nan, nan, nan, nan);
            }
            const double nd = static_cast<double>(n_);
            const double Sx = c.st.sx, Sy = c.st.sy;
            const double Sxx = c.st.sxx, Syy = c.st.syy, Sxy = c.st.sxy;

            const double nxx = nd * Sxx, sx2 = Sx * Sx;
            const double denom_x = nxx - sx2;
            if (denom_x <= 0.0) {
                return std::make_tuple(nan, nan, nan, nan);
            }
            const double nxy = nd * Sxy, sxsy = Sx * Sy;
            const double num = nxy - sxsy;
            const double slope = num / denom_x;
            // mean_y - slope * mean_x, with the means restored from the origins.
            const double mean_x = c.st.ox + Sx / nd;
            const double mean_y = c.st.oy + Sy / nd;
            const double slope_mean_x = slope * mean_x;
            const double intercept = mean_y - slope_mean_x;

            const double nyy = nd * Syy, sy2 = Sy * Sy;
            const double denom_y = nyy - sy2;
            const double num2 = num * num;
            const double dxy = denom_x * denom_y;
            const double r_sq = (denom_y > 0.0) ? num2 / dxy : nan;

            // SST = Syy - Sy^2/n; SSR = slope * (Sxy - Sx*Sy/n); SSE = SST - SSR.
            const double SST = Syy - Sy * Sy / nd;
            const double SSR = slope * (Sxy - Sx * Sy / nd);
            const double SSE = SST - SSR;
            const double stderr_val = (SSE > 0.0)
                                    ? std::sqrt(SSE / (nd - 2.0))
                                    : 0.0;
            return std::make_tuple(slope, intercept, r_sq, stderr_val);
        }
    };
    Stepper stepper() {
        return Stepper{sums_.cursor(), start_policy_ == detail::StartPolicy::Strict};
    }
    void store(const Stepper& k) { sums_.set_state(k.c.st); }

    ResultTuple call(const InputArray& inputs) override {
        Stepper k = stepper();
        const ResultTuple r = k.call(inputs);
        store(k);
        return r;
    }

private:
    const detail::StartPolicy start_policy_;
    Sums sums_;
};

}  // namespace screamer

#endif
