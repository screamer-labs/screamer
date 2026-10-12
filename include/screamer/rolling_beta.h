#ifndef SCREAMER_ROLLING_BETA_H
#define SCREAMER_ROLLING_BETA_H

// RollingBeta: rolling regression slope of x on y.
//
//     beta_w[t] = cov(x, y) / var(y)
//               = ( n*Sxy - Sx*Sy ) / ( n*Syy - Sy^2 )
//
// (the n*(n-1) factors in cov and var cancel, so we can use the
// uncentered sums directly)
//
// Convention: the FIRST argument is the dependent (target), the SECOND
// is the regressor. Calling RollingBeta(asset_returns, market_returns, w)
// returns the asset's beta with respect to the market, matching the CAPM
// definition. Pandas uses the opposite convention (slope of y on x).
//
// Window sums about an origin near the means (detail::ShiftedWindowCoSums),
// so the result stays accurate when the levels are large next to the spread.
// O(1) per step.

#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include "screamer/common/functor_base.h"
#include "screamer/common/float_info.h"
#include "screamer/detail/shifted_window_sums.h"
#include "screamer/detail/start_policy.h"
#include <cstddef>

namespace screamer {

class RollingBeta : public FunctorBase<RollingBeta, 2, 1> {
    using Sums = detail::ShiftedWindowCoSums<false, true>;

public:
    RollingBeta(int window_size,
          const std::string& start_policy = "strict")
        : start_policy_(detail::parse_start_policy(start_policy)),
          sums_(window_size >= 2 ? window_size : 2, start_policy_)
    {
        if (window_size < 2) {
            throw std::invalid_argument("Window size must be 2 or more.");
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
            const double x = inputs[0];
            const double y = inputs[1];
            if (isnan2(x) || isnan2(y)) {
                return std::numeric_limits<double>::quiet_NaN();
            }
            c.push(x, y);
            const std::size_t n_ = c.st.count;
            if ((strict && n_ < c.cap) || n_ < 2) {
                return std::numeric_limits<double>::quiet_NaN();
            }
            const double nd = static_cast<double>(n_);
            const double a = nd * c.st.sxy;
            const double b = c.st.sx * c.st.sy;
            const double num = a - b;
            const double ayy = nd * c.st.syy;
            const double byy = c.st.sy * c.st.sy;
            const double denom = ayy - byy;
            if (denom <= 0.0) {
                return std::numeric_limits<double>::quiet_NaN();
            }
            return num / denom;
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

#endif  // SCREAMER_ROLLING_BETA_H
