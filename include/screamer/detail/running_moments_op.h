#ifndef SCREAMER_DETAIL_RUNNING_MOMENTS_OP_H
#define SCREAMER_DETAIL_RUNNING_MOMENTS_OP_H

// The shared base of the one-input expanding (kDecay = false) and
// exponentially weighted (kDecay = true) moment operators, and of
// ExpandingSlope (kTime = true). It owns the power
// sums (detail::ShiftedPowerSums<P, kTime>) and, for the weighted ones, the sum of
// squared weights, and runs the per-event and batch loops; Derived supplies
//
//     double value(const State& st, double x) const
//
// the output for the state st after adding x. The batch loop keeps the state
// in a local, and both loops call the same step and value, so batch and
// streaming give the same numbers. A NaN input gives NaN and leaves the state
// untouched (the "ignore" NaN policy).

#include <cstddef>
#include <limits>
#include "screamer/common/base.h"
#include "screamer/common/float_info.h"
#include "screamer/detail/shifted_power_sums.h"

namespace screamer {
namespace detail {

template <class Derived, int P, bool kDecay, bool kTime = false>
class RunningMomentsOp : public ScreamerBase {
    static_assert(!(kDecay && kTime), "time sums are for the expanding operators");

protected:
    struct State {
        ShiftedPowerSums<P, kTime> sums;  // w = sum of the weights
        double w2 = 0.0;           // sum of the squared weights (kDecay only)
    };

public:
    void reset() override { st_ = State{}; }

    double process_scalar(double x) override {
        if (isnan2(x)) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        step(st_, x);
        return static_cast<const Derived&>(*this).value(st_, x);
    }

    void process_array_no_stride(double* y, const double* x, size_t size) override {
        run(y, 1, x, 1, size);
    }

    void process_array_stride(double* y, size_t dy, const double* x, size_t dx,
                              size_t size) override {
        run(y, dy, x, dx, size);
    }

protected:
    // Decays every weight by (1 - alpha) (kDecay), then adds x with weight 1.
    inline void step(State& st, double x) const {
        if constexpr (kDecay) {
            st.sums.decay(one_minus_alpha_);
            st.w2 *= one_minus_alpha2_;
        }
        st.sums.add(x);
        if constexpr (kDecay) {
            st.w2 += 1.0;
        }
    }

    // Set by the weighted operators' constructors.
    double one_minus_alpha_ = 1.0;
    double one_minus_alpha2_ = 1.0;

private:
    void run(double* y, size_t dy, const double* x, size_t dx, size_t size) {
        const Derived& self = static_cast<const Derived&>(*this);
        State st = st_;
        for (size_t i = 0; i < size; ++i, y += dy, x += dx) {
            const double v = *x;
            if (isnan2(v)) {
                *y = std::numeric_limits<double>::quiet_NaN();
                continue;
            }
            step(st, v);
            *y = self.value(st, v);
        }
        st_ = st;
    }

    State st_;
};

}  // namespace detail
}  // namespace screamer

#endif  // SCREAMER_DETAIL_RUNNING_MOMENTS_OP_H
