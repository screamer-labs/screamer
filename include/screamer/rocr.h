#ifndef SCREAMER_ROCR_H
#define SCREAMER_ROCR_H

// ROCR(k): rate of change as a ratio,
//
//     ROCR[t] = x[t] / x[t-k]
//
// TA-Lib's ROCR. NaN for the first k samples and when x[t-k] == 0.

#include <limits>
#include <stdexcept>
#include "screamer/common/base.h"
#include "screamer/common/buffer.h"

namespace screamer {

class ROCR : public ScreamerBase {
public:
    explicit ROCR(int window_size)
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
        return newValue / oldValue;
    }

private:
    const int window_size_;
    FixedSizeBuffer buffer_;
};

}  // namespace screamer

#endif
