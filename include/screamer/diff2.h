#ifndef SCREAMER_DIFF2_H
#define SCREAMER_DIFF2_H

// Diff2: second-order finite difference (discrete second derivative).
//
//     y[t] = x[t] - 2*x[t-1] + x[t-2]
//          = (x[t] - x[t-1]) - (x[t-1] - x[t-2])
//
// Implemented as two chained lag-1 delays, so the warmup naturally
// emits NaN for the first two samples under start_policy="strict".
// This is *not* the same as Diff(2), which is the lag-2 first
// difference x[t] - x[t-2].

#include <string>
#include "screamer/common/base.h"
#include <cstddef>
#include <limits>
#include "screamer/detail/start_policy.h"

namespace screamer {

class Diff2 : public ScreamerBase {
public:
    Diff2(const std::string& start_policy = "strict")
        : policy_(detail::parse_start_policy(start_policy))
    {}

    void reset() override { state_ = State{}; }

    double process_scalar(double x) override { return step(state_, x); }

    void process_array_no_stride(double* y, const double* x, size_t size) override {
        run_column(y, 1, x, 1, size);
    }

    void process_array_stride(double* y, size_t dyi, const double* x, size_t dxi,
                              size_t size) override {
        run_column(y, static_cast<std::ptrdiff_t>(dyi), x, static_cast<std::ptrdiff_t>(dxi), size);
    }

private:
    // The two lag-1 delay lines: the last value of each and whether it has
    // one yet. NaN is not skipped; it flows through like any value.
    struct State {
        double last1 = std::numeric_limits<double>::quiet_NaN();
        double last2 = std::numeric_limits<double>::quiet_NaN();
        bool full1 = false;
        bool full2 = false;
    };

    // A lag-1 delay line (detail::DelayBuffer of size 1): returns the previous
    // value, or during warmup what the start policy gives.
    double delay(double& last, bool& full, double v) const {
        const double old = last;
        last = v;
        if (full) return old;
        full = true;
        if (policy_ == detail::StartPolicy::Strict) return std::numeric_limits<double>::quiet_NaN();
        if (policy_ == detail::StartPolicy::Zero) return 0.0;
        return v;
    }

    double step(State& s, double x) const {
        const double d1 = x - delay(s.last1, s.full1, x);
        return d1 - delay(s.last2, s.full2, d1);
    }

    // The batch loop, on a local copy of the state.
    void run_column(double* y, std::ptrdiff_t dy, const double* x, std::ptrdiff_t dx,
                    size_t size) {
        State s = state_;
        for (size_t m = 0; m < size; ++m, x += dx, y += dy) *y = step(s, *x);
        state_ = s;
    }

    detail::StartPolicy policy_;
    State state_;
};

}  // namespace screamer

#endif  // SCREAMER_DIFF2_H
