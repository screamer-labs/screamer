#ifndef SCREAMER_PROPAGATOR_H
#define SCREAMER_PROPAGATOR_H

#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>
#include <algorithm>
#include "screamer/common/base.h"

namespace screamer {

    // Bouchaud-Gefen-Potters-Wyart (2004) propagator model: price impact as a
    // decaying-kernel convolution over past signed order flow,
    //     impact_t = sum_{k=0}^{window-1} G(k) * flow_{t-k},   G(k) = g0 * (k+1)^(-gamma).
    // Flow moves price with a memory that decays but does not vanish at once.
    // This is a positional (FIR) filter, so it follows the "propagate" NaN
    // policy: a NaN flow is kept in the window and flows through the convolution
    // (the output is NaN while the NaN is inside the window and recovers once it
    // leaves), exactly as Lag/Diff propagate. The first window-1 samples are NaN
    // (warmup, the window is not yet full).
    class Propagator : public ScreamerBase {
    public:
        Propagator(int window = 20, double g0 = 1.0, double gamma = 0.5)
            : window_(window)
        {
            if (window_ < 1) {
                throw std::invalid_argument("window must be at least 1");
            }
            kernel_.resize(window_);
            for (int k = 0; k < window_; ++k) {
                kernel_[k] = g0 * std::pow(static_cast<double>(k) + 1.0, -gamma);
            }
            buffer_.resize(window_);
            reset();
        }

        void reset() override {
            index_ = 0;
            size_ = 0;
            std::fill(buffer_.begin(), buffer_.end(), 0.0);
        }

        double process_scalar(double flow) override {
            // "propagate": store every value, including NaN, faithfully.
            buffer_[index_] = flow;
            index_ = (index_ + 1) % window_;

            if (size_ < window_) {
                size_++;
                if (size_ < window_) {
                    return std::numeric_limits<double>::quiet_NaN();   // warmup
                }
            }

            // The just-written newest sample (flow_t, k=0) sits one slot back
            // from index_; flow_{t-k} is k slots back from there. A NaN in the
            // window propagates through the sum via IEEE arithmetic.
            const int newest = (index_ - 1 + window_) % window_;
            double acc = 0.0;
            for (int k = 0; k < window_; ++k) {
                const int pos = (newest - k + window_) % window_;
                acc += kernel_[k] * buffer_[pos];
            }
            return acc;
        }

    private:
        const int window_;
        int index_ = 0;
        int size_ = 0;
        std::vector<double> kernel_;
        std::vector<double> buffer_;
    };

} // namespace screamer

#endif // SCREAMER_PROPAGATOR_H
