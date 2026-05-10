#ifndef SCREAMER_ROLLING_MIN_MAX_H
#define SCREAMER_ROLLING_MIN_MAX_H

// RollingMinMax: rolling minimum and maximum of a single stream over a
// fixed window. Returns the pair (min, max) per step. Useful when both
// extrema are needed (e.g. for normalised range, channels, breakouts);
// computing them in one pass is roughly half the work of running
// RollingMin and RollingMax independently.
//
// Algorithm: two monotonic deques, identical to the one in RollingMin
// and RollingMax. The min deque keeps elements in non-decreasing order
// from front to back; its front is the current rolling minimum. The max
// deque keeps elements in non-increasing order; its front is the
// current rolling maximum. Both updates are amortised O(1) per step.

#include <deque>
#include <stdexcept>
#include <tuple>
#include <utility>
#include "screamer/common/functor_base.h"

namespace screamer {

class RollingMinMax : public FunctorBase<RollingMinMax, 1, 2> {
public:
    explicit RollingMinMax(int window_size) : window_size_(window_size) {
        if (window_size <= 0) {
            throw std::invalid_argument("Window size must be positive.");
        }
        reset();
    }

    void reset() override {
        min_deque.clear();
        max_deque.clear();
        index = 0;
    }

    ResultTuple call(const InputArray& inputs) override {
        const double newValue = inputs[0];

        // Min deque: drop trailing elements >= newValue (they cannot be
        // the rolling min for any future window that contains newValue).
        while (!min_deque.empty() && min_deque.back().first >= newValue) {
            min_deque.pop_back();
        }
        min_deque.emplace_back(newValue, index);
        if (min_deque.front().second <= index - window_size_) {
            min_deque.pop_front();
        }

        // Max deque: dual of the above.
        while (!max_deque.empty() && max_deque.back().first <= newValue) {
            max_deque.pop_back();
        }
        max_deque.emplace_back(newValue, index);
        if (max_deque.front().second <= index - window_size_) {
            max_deque.pop_front();
        }

        index++;
        return std::make_tuple(min_deque.front().first, max_deque.front().first);
    }

private:
    int window_size_;
    int index = 0;
    std::deque<std::pair<double, int>> min_deque;
    std::deque<std::pair<double, int>> max_deque;
};

}  // namespace screamer

#endif  // SCREAMER_ROLLING_MIN_MAX_H
