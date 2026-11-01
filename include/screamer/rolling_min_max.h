#ifndef SCREAMER_ROLLING_MIN_MAX_H
#define SCREAMER_ROLLING_MIN_MAX_H

// RollingMinMax: rolling minimum and maximum of a single stream over a
// fixed window. Returns the pair (min, max) per step. Useful when both
// extrema are needed (e.g. for normalised range, channels, breakouts);
// computing them in one pass is roughly half the work of running
// RollingMin and RollingMax independently.
//
// Algorithm: two monotonic deques (detail::MinDeque, detail::MaxDeque),
// the same primitive used by RollingMin and RollingMax. Amortised O(1)
// per step.

#include <cmath>
#include <limits>
#include <tuple>
#include "screamer/common/functor_base.h"
#include "screamer/detail/monotonic_deque.h"

namespace screamer {

class RollingMinMax : public FunctorBase<RollingMinMax, 1, 2> {
public:
    static constexpr const char* kTypeName = "RollingMinMax";
    static constexpr int kStateVersion = 1;

    template <class Ar>
    void settings(Ar& ar) {
        ar("window_size", min_deque_.window_size());
    }

    template <class Ar>
    void state(Ar& ar) {
        ar.object("min", min_deque_);
        ar.object("max", max_deque_);
        ar.check(min_deque_.samples_seen() == max_deque_.samples_seen(),
                 "min and max must have seen the same samples");
    }

    explicit RollingMinMax(int window_size)
        : min_deque_(window_size), max_deque_(window_size) {}

    void reset() override {
        min_deque_.reset();
        max_deque_.reset();
    }

    ResultTuple call(const InputArray& inputs) override {
        const double newValue = inputs[0];
        if (std::isnan(newValue)) {
            const double nan = std::numeric_limits<double>::quiet_NaN();
            return std::make_tuple(nan, nan);
        }
        const double current_min = min_deque_.append(newValue);
        const double current_max = max_deque_.append(newValue);
        return std::make_tuple(current_min, current_max);
    }

private:
    detail::MinDeque min_deque_;
    detail::MaxDeque max_deque_;
};

}  // namespace screamer

#endif  // SCREAMER_ROLLING_MIN_MAX_H
