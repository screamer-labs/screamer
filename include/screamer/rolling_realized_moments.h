#ifndef SCREAMER_ROLLING_REALIZED_MOMENTS_H
#define SCREAMER_ROLLING_REALIZED_MOMENTS_H

// RollingRealizedSkew and RollingRealizedKurt: the realized skewness and
// kurtosis of a window of N returns (Amaya, Christoffersen, Jacobs and
// Vasquez 2015):
//
//     RV    = sum r_i^2
//     RSkew = sqrt(N) * sum r_i^3 / RV^(3/2)
//     RKurt = N * sum r_i^4 / RV^2
//
// The moments are taken about zero, not about the window mean, and carry no
// small-sample correction: the mean of high-frequency returns is negligible
// next to their spread, and the paper's measures are these ratios of power
// sums. RKurt is the plain kurtosis (3 for normal returns), not the excess.
// A window whose returns are all zero has RV = 0 and gives NaN.
//
// The window keeps its returns in a ring and the power sums up to date by
// adding the new return and subtracting the one that leaves. The sums are
// recomputed from the ring when the return that left dominates what remains
// (its square above K times RV, or for the fourth power its r^4 above K times
// sum r^4, K = 16), when a non-finite return has left, and once every
// window_size steps, as in detail::ShiftedWindowSums. Without the recompute,
// a large return that left would leave rounding of its own size in the
// smaller sums; with it, a window of zeros after a spike gives exactly RV = 0.
// The cost is O(1) amortized per step.
//
// start_policy: "strict" gives NaN until the window holds window_size returns,
// "expanding" uses the returns seen so far (N = their count), "zero" starts
// from a window of window_size zeros (N = window_size). nan_policy: ignore.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
#include "screamer/common/base.h"
#include "screamer/common/float_info.h"
#include "screamer/common/functor_base.h"   // SCREAMER_NOINLINE
#include "screamer/detail/start_policy.h"

namespace screamer {
namespace detail {

// The window of returns with sum r^2 and sum r^P (P = 3 or 4). Derived
// supplies double value(const State&) const for the current window.
template <class Derived, int P>
class RealizedMomentsOp : public ScreamerBase {
    static_assert(P == 3 || P == 4, "RealizedMomentsOp: P is 3 or 4");

protected:
    struct State {
        double s2 = 0.0;          // sum r^2
        double sp = 0.0;          // sum r^P
        std::size_t count = 0;    // returns in the window
        std::size_t pos = 0;      // next slot of the ring
        std::size_t since = 0;    // steps since the last recompute
    };

    RealizedMomentsOp(int window_size, const std::string& start_policy)
        : policy_(parse_start_policy(start_policy)),
          buf_(window_size >= 2 ? static_cast<std::size_t>(window_size) : 1, 0.0)
    {
        if (window_size < 2) throw std::invalid_argument("Window size must be 2 or more.");
        reset();
    }

public:
    void reset() override {
        std::fill(buf_.begin(), buf_.end(), 0.0);
        st_ = State{};
        if (policy_ == StartPolicy::Zero) st_.count = buf_.size();
    }

    double process_scalar(double x) override {
        if (isnan2(x)) return std::numeric_limits<double>::quiet_NaN();
        State st = st_;
        step(st, buf_.data(), buf_.size(), x);
        st_ = st;
        return static_cast<const Derived&>(*this).value(st);
    }

    void process_array_no_stride(double* y, const double* x, size_t size) override {
        run(y, 1, x, 1, size);
    }

    void process_array_stride(double* y, size_t dy, const double* x, size_t dx,
                              size_t size) override {
        run(y, dy, x, dx, size);
    }

protected:
    // N for the current window, or 0 where the policy gives no output.
    double n_or_zero(const State& st) const {
        if (policy_ == StartPolicy::Strict && st.count != buf_.size()) return 0.0;
        return static_cast<double>(st.count);
    }

private:
    static double power_p(double r, double r2) {
        if constexpr (P == 3) {
            const double r3 = r2 * r;
            return r3;
        } else {
            const double r4 = r2 * r2;
            return r4;
        }
    }

    // One finite return into the window. Each product is its own statement,
    // so the step rounds alike inlined into the batch loop or called alone
    // (see shifted_window_sums.h).
    static void step(State& st, double* buf, std::size_t cap, double r) {
        constexpr double kTerm = 16.0;
        const double r2 = r * r;
        const double rp = power_p(r, r2);
        bool recompute = false;
        if (st.count == cap) {
            const double o = buf[st.pos];
            const double o2 = o * o;
            const double op = power_p(o, o2);
            // One add on each running sum's dependency chain per step.
            const double d2 = r2 - o2;
            const double dp = rp - op;
            st.s2 = st.s2 + d2;
            st.sp = st.sp + dp;
            if (!std::isfinite(o)) {
                recompute = true;
            } else {
                const double k2 = kTerm * st.s2;
                recompute = o2 > k2;
                if constexpr (P == 4) {
                    const double k4 = kTerm * st.sp;
                    recompute = recompute || op > k4;
                }
            }
        } else {
            ++st.count;
            st.s2 = st.s2 + r2;
            st.sp = st.sp + rp;
        }
        buf[st.pos] = r;
        if (++st.pos == cap) st.pos = 0;
        if (++st.since >= cap) recompute = true;
        if (recompute) resum(st, buf, cap);
    }

    // The exact sums from the ring. It runs about once per window; kept out
    // of line it leaves the step small enough for the batch loop to keep its
    // state in registers.
    // Four partial sums per power break the add chain; the order is fixed,
    // so every caller gets the same rounding.
    SCREAMER_NOINLINE static void resum(State& st, const double* buf, std::size_t cap) {
        double s2[4] = {0.0, 0.0, 0.0, 0.0};
        double sp[4] = {0.0, 0.0, 0.0, 0.0};
        std::size_t i = 0;
        for (; i + 4 <= cap; i += 4) {
            for (std::size_t j = 0; j < 4; ++j) {
                const double v = buf[i + j];
                const double v2 = v * v;
                const double vp = power_p(v, v2);
                s2[j] = s2[j] + v2;
                sp[j] = sp[j] + vp;
            }
        }
        for (std::size_t j = 0; i < cap; ++i, ++j) {
            const double v = buf[i];
            const double v2 = v * v;
            const double vp = power_p(v, v2);
            s2[j] = s2[j] + v2;
            sp[j] = sp[j] + vp;
        }
        st.s2 = (s2[0] + s2[1]) + (s2[2] + s2[3]);
        st.sp = (sp[0] + sp[1]) + (sp[2] + sp[3]);
        st.since = 0;
    }

    void run(double* y, size_t dy, const double* x, size_t dx, size_t size) {
        const Derived& self = static_cast<const Derived&>(*this);
        double* buf = buf_.data();
        const std::size_t cap = buf_.size();
        State st = st_;
        for (size_t i = 0; i < size; ++i, y += dy, x += dx) {
            const double v = *x;
            if (isnan2(v)) {
                *y = std::numeric_limits<double>::quiet_NaN();
                continue;
            }
            step(st, buf, cap, v);
            *y = self.value(st);
        }
        st_ = st;
    }

    const StartPolicy policy_;
    std::vector<double> buf_;   // unused slots hold 0, so a recompute sums the whole ring
    State st_;
};

}  // namespace detail

class RollingRealizedSkew
    : public detail::RealizedMomentsOp<RollingRealizedSkew, 3> {
public:
    RollingRealizedSkew(int window_size = 20, const std::string& start_policy = "strict")
        : RealizedMomentsOp(window_size, start_policy) {}

    double value(const State& st) const {
        const double n = n_or_zero(st);
        if (n == 0.0 || !(st.s2 > 0.0)) return std::numeric_limits<double>::quiet_NaN();
        return std::sqrt(n) * st.sp / (st.s2 * std::sqrt(st.s2));
    }
};

class RollingRealizedKurt
    : public detail::RealizedMomentsOp<RollingRealizedKurt, 4> {
public:
    RollingRealizedKurt(int window_size = 20, const std::string& start_policy = "strict")
        : RealizedMomentsOp(window_size, start_policy) {}

    double value(const State& st) const {
        const double n = n_or_zero(st);
        if (n == 0.0 || !(st.s2 > 0.0)) return std::numeric_limits<double>::quiet_NaN();
        return n * st.sp / (st.s2 * st.s2);
    }
};

}  // namespace screamer

#endif  // SCREAMER_ROLLING_REALIZED_MOMENTS_H
