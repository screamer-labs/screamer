#ifndef SCREAMER_ROLLING_TLS_BETA_H
#define SCREAMER_ROLLING_TLS_BETA_H

// RollingTLSBeta: rolling total least squares (orthogonal regression) slope
// of the first input on the second.
//
// Ordinary least squares puts all the noise in the dependent series, so the
// slope of a on b is not the inverse of the slope of b on a, and a pairs
// trade's hedge ratio depends on which leg is called dependent. Total least
// squares minimizes the perpendicular distance to the line instead, which
// treats both series alike: the slope of a on b is exactly 1 / the slope of b
// on a. With centered sums Saa, Sbb, Sab over the window,
//
//     beta = (Saa - Sbb + sqrt((Saa - Sbb)^2 + 4 Sab^2)) / (2 Sab)
//
// It assumes equal noise in both series, so use it on series of the same
// scale, such as log prices. NaN when Sab = 0 (no linear relation). Same
// argument order as RollingBeta: the first input is the dependent series.
//
// Window sums about an origin near the means (detail::ShiftedWindowCoSums),
// so the slope stays accurate on price levels. O(1) per step.

#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include "screamer/common/functor_base.h"
#include "screamer/common/float_info.h"
#include "screamer/detail/moment_formulas.h"
#include "screamer/detail/shifted_window_sums.h"
#include "screamer/detail/start_policy.h"

namespace screamer {

class RollingTLSBeta : public StepperFunctorBase<RollingTLSBeta, 2, 1> {
    using Sums = detail::ShiftedWindowCoSums<>;

public:
    RollingTLSBeta(int window_size, const std::string& start_policy = "strict")
        : start_policy_(detail::parse_start_policy(start_policy)),
          sums_(window_size >= 2 ? window_size : 2, start_policy_)
    {
        if (window_size < 2) {
            throw std::invalid_argument("Window size must be 2 or more.");
        }
        reset();
    }

    void reset() override { sums_.reset(); }

    // One event on the window's state, for both the batch column loop and
    // call() (StepperFunctorBase). Each product is its own statement (see
    // shifted_window_sums.h).
    struct Stepper {
        Sums::Cursor c;
        bool strict;

        inline ResultTuple call(const InputArray& inputs) {
            const double a = inputs[0];
            const double b = inputs[1];
            if (isnan2(a) || isnan2(b)) {
                return std::numeric_limits<double>::quiet_NaN();
            }
            c.push(a, b);
            const std::size_t n_ = c.st.count;
            if ((strict && n_ < c.cap) || n_ < 2) {
                return std::numeric_limits<double>::quiet_NaN();
            }
            const double nd = static_cast<double>(n_);
            // With x = a and y = b.
            const double caa = detail::n2_comoment(c.st.sx, c.st.sx, c.st.sxx, nd);
            const double cbb = detail::n2_comoment(c.st.sy, c.st.sy, c.st.syy, nd);
            const double cab = detail::n2_comoment(c.st.sx, c.st.sy, c.st.sxy, nd);
            if (cab == 0.0) return std::numeric_limits<double>::quiet_NaN();
            const double d = caa - cbb;
            const double dd = d * d;
            const double cab2 = 4.0 * cab * cab;
            return (d + std::sqrt(dd + cab2)) / (2.0 * cab);
        }
    };
    Stepper stepper() {
        return Stepper{sums_.cursor(), start_policy_ == detail::StartPolicy::Strict};
    }
    void store(const Stepper& k) { sums_.set_state(k.c.st); }

private:
    const detail::StartPolicy start_policy_;
    Sums sums_;
};

}  // namespace screamer

#endif
