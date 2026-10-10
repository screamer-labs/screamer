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
// Five detail::RollingSum buffers, O(1) per step.

#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include "screamer/common/functor_base.h"
#include "screamer/common/float_info.h"
#include "screamer/detail/rolling_sum.h"
#include "screamer/detail/start_policy.h"

namespace screamer {

class RollingTLSBeta : public FunctorBase<RollingTLSBeta, 2, 1> {
public:
    RollingTLSBeta(int window_size, const std::string& start_policy = "strict")
        : window_size_(window_size),
          start_policy_(detail::parse_start_policy(start_policy)),
          sa_(window_size, "expanding"), sb_(window_size, "expanding"),
          saa_(window_size, "expanding"), sbb_(window_size, "expanding"),
          sab_(window_size, "expanding")
    {
        if (window_size < 2) throw std::invalid_argument("Window size must be 2 or more.");
        reset();
    }

    void reset() override {
        sa_.reset(); sb_.reset(); saa_.reset(); sbb_.reset(); sab_.reset();
        n_ = (start_policy_ == detail::StartPolicy::Zero) ? window_size_ : 0;
    }

    ResultTuple call(const InputArray& in) override {
        const double a = in[0], b = in[1];
        const double nan = std::numeric_limits<double>::quiet_NaN();
        if (isnan2(a) || isnan2(b)) return nan;
        const double sa = sa_.append(a), sb = sb_.append(b);
        const double saa = saa_.append(a * a), sbb = sbb_.append(b * b), sab = sab_.append(a * b);
        if (n_ < static_cast<std::size_t>(window_size_) &&
            start_policy_ != detail::StartPolicy::Zero) ++n_;
        if (start_policy_ == detail::StartPolicy::Strict &&
            n_ < static_cast<std::size_t>(window_size_)) return nan;
        if (n_ < 2) return nan;
        const double n = static_cast<double>(n_);
        const double caa = n * saa - sa * sa, cbb = n * sbb - sb * sb, cab = n * sab - sa * sb;
        if (cab == 0.0) return nan;
        const double d = caa - cbb;
        return (d + std::sqrt(d * d + 4.0 * cab * cab)) / (2.0 * cab);
    }

private:
    const int window_size_;
    const detail::StartPolicy start_policy_;
    detail::RollingSum sa_, sb_, saa_, sbb_, sab_;
    std::size_t n_ = 0;
};

}  // namespace screamer

#endif
