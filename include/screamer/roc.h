#ifndef SCREAMER_ROC_H
#define SCREAMER_ROC_H

// ROC(k): rate of change as a percentage,
//
//     ROC[t] = 100 * (x[t] - x[t-k]) / x[t-k]
//
// Same as 100 * Return(k); separately named to match TA-Lib's ROC.
// Returns NaN for the first k samples (no x[t-k] yet) and when
// x[t-k] == 0.

#include <limits>
#include <stdexcept>
#include "screamer/common/base.h"
#include "screamer/common/buffer.h"

namespace screamer {

class ROC : public ScreamerBase {
public:
    explicit ROC(int window_size)
        : window_size_(window_size),
          buffer_(window_size, std::numeric_limits<double>::quiet_NaN())
    {
        if (window_size <= 0) {
            throw std::invalid_argument("Window size must be positive.");
        }
    }

    void reset() override {
        buffer_.reset(std::numeric_limits<double>::quiet_NaN());
    }

    double process_scalar(double newValue) override {
        const double oldValue = buffer_.append(newValue);
        return 100.0 * (newValue - oldValue) / oldValue;
    }

private:
    const int window_size_;
    FixedSizeBuffer buffer_;
};

}  // namespace screamer

#endif
