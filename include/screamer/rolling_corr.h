#ifndef SCREAMER_ROLLING_CORR_H
#define SCREAMER_ROLLING_CORR_H

// RollingCorr: rolling Pearson correlation of two streams.
//
// Math:
//     r = ( n*Sxy - Sx*Sy ) / sqrt( (n*Sxx - Sx^2) * (n*Syy - Sy^2) )
//
// Where Sx, Sy, Sxx, Syy, Sxy are the rolling sums over the window.
// Window sums about an origin near the means (detail::ShiftedWindowCoSums)
// keep it accurate when the levels are large next to the spread; O(1) per step.
//
// This is the first real algorithm built on FunctorBase<_, 2, 1> (two
// inputs, one output). It validates the multi-input dispatch path that's
// already implemented in functor_base.h's handle_input_Ni_1o family.

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

class RollingCorr : public FunctorBase<RollingCorr, 2, 1> {
    using Sums = detail::ShiftedWindowCoSums<true, true>;

public:
    RollingCorr(int window_size,
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
            const double cov = a - b;
            const double axx = nd * c.st.sxx;
            const double bxx = c.st.sx * c.st.sx;
            const double var_x = axx - bxx;
            const double ayy = nd * c.st.syy;
            const double byy = c.st.sy * c.st.sy;
            const double var_y = ayy - byy;
            const double denom2 = var_x * var_y;
            if (denom2 <= 0.0) {
                return std::numeric_limits<double>::quiet_NaN();
            }
            return cov / std::sqrt(denom2);
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

#endif  // SCREAMER_ROLLING_CORR_H
