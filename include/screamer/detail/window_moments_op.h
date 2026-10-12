#ifndef SCREAMER_DETAIL_WINDOW_MOMENTS_OP_H
#define SCREAMER_DETAIL_WINDOW_MOMENTS_OP_H

// The shared base of the one-input rolling moment operators (RollingVar,
// RollingStd, RollingSkew, RollingKurt, RollingZscore). It owns the window
// (detail::ShiftedWindowSums<P>) and the per-event and batch loops; Derived
// supplies
//
//     double value(const State& st, double x) const
//
// the output for the window state st after adding x. The batch loop keeps the
// state in a local, and both loops call the same step and value, so batch and
// streaming give the same numbers. A NaN input gives NaN and leaves the window
// untouched (the "ignore" NaN policy).

#include <cstddef>
#include <limits>
#include <string>
#include "screamer/common/base.h"
#include "screamer/common/float_info.h"
#include "screamer/detail/shifted_window_sums.h"
#include "screamer/detail/start_policy.h"

namespace screamer {
namespace detail {

template <class Derived, int P>
class WindowMomentsOp : public ScreamerBase {
protected:
    using Sums = ShiftedWindowSums<P>;
    using State = typename Sums::State;

    // window_size is validated by Derived, which throws for a size it does not
    // accept; the window is built with at least one slot so that the check is
    // reached.
    WindowMomentsOp(int window_size, StartPolicy start_policy, StartPolicy sums_policy)
        : window_size_(window_size),
          start_policy_(start_policy),
          sums_(window_size > 0 ? static_cast<std::size_t>(window_size) : 1, sums_policy) {}

public:
    void reset() override { sums_.reset(); }

    double process_scalar(double x) override {
        if (isnan2(x)) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        typename Sums::Cursor c = sums_.cursor();
        c.push(x);
        sums_.set_state(c.st);
        return static_cast<const Derived&>(*this).value(c.st, x);
    }

    void process_array_no_stride(double* y, const double* x, size_t size) override {
        run(y, 1, x, 1, size);
    }

    void process_array_stride(double* y, size_t dy, const double* x, size_t dx,
                              size_t size) override {
        run(y, dy, x, dx, size);
    }

private:
    void run(double* y, size_t dy, const double* x, size_t dx, size_t size) {
        const Derived& self = static_cast<const Derived&>(*this);
        typename Sums::Cursor c = sums_.cursor();
        for (size_t i = 0; i < size; ++i, y += dy, x += dx) {
            const double v = *x;
            if (isnan2(v)) {
                *y = std::numeric_limits<double>::quiet_NaN();
                continue;
            }
            c.push(v);
            *y = self.value(c.st, v);
        }
        sums_.set_state(c.st);
    }

protected:
    const int window_size_;
    const StartPolicy start_policy_;
    Sums sums_;
};

}  // namespace detail
}  // namespace screamer

#endif  // SCREAMER_DETAIL_WINDOW_MOMENTS_OP_H
