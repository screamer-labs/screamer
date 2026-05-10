#ifndef SCREAMER_ROLLING_MIN_H
#define SCREAMER_ROLLING_MIN_H

#include "screamer/common/base.h"
#include "screamer/detail/monotonic_deque.h"

namespace screamer {

class RollingMin : public ScreamerBase {
public:
    explicit RollingMin(int window_size) : deque_(window_size) {}

    void reset() override { deque_.reset(); }

private:
    double process_scalar(double newValue) override {
        return deque_.append(newValue);
    }

    detail::MinDeque deque_;
};

}  // namespace screamer

#endif
