#ifndef SCREAMER_ROLLING_COV_H
#define SCREAMER_ROLLING_COV_H

// RollingCov: rolling sample covariance of two streams.
//
//     cov_w[t] = ( n*Sxy - Sx*Sy ) / ( n*(n-1) )
//
// where Sx, Sy, Sxy are the rolling sums over the window and n = window_size
// once warmup is complete. The (n-1) denominator gives the unbiased sample
// estimate, matching pandas Series.rolling(w).cov(other).
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

class RollingCov : public FunctorBase<RollingCov, 2, 1> {
    using Sums = detail::ShiftedWindowCoSums<false, false>;

public:
    RollingCov(int window_size, const std::string& start_policy = "strict")
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
            const double d = nd * (nd - 1.0);
            return (a - b) / d;
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

#endif  // SCREAMER_ROLLING_COV_H
