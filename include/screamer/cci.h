#ifndef SCREAMER_CCI_H
#define SCREAMER_CCI_H

// CCI: Commodity Channel Index (Donald Lambert).
//
//     TP[t]  = (high + low + close) / 3
//     SMA_TP = SMA(TP, n)
//     MAD    = mean( |TP - SMA_TP| ) over the same window
//     CCI[t] = (TP[t] - SMA_TP[t]) / (0.015 * MAD[t])
//
// 3 -> 1 on (high, low, close). The 0.015 scaling is a Lambert
// convention: roughly 70-80% of CCI values fall in [-100, +100]
// for a normal distribution input.
//
// Composition: detail::RollingMean for SMA(TP), and a circular buffer
// + per-step MAD calculation (same algorithm as RollingMad). The
// SMA buffer and the MAD's window share the same period, so we keep
// one buffer and derive both quantities.

#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>
#include "screamer/common/functor_base.h"

namespace screamer {

class CCI : public FunctorBase<CCI, 3, 1> {
public:
    explicit CCI(int window_size = 14)
        : window_size_(window_size)
    {
        if (window_size < 2) {
            throw std::invalid_argument("Window size must be at least 2.");
        }
        buffer_.resize(window_size_);
        reset();
    }

    void reset() override {
        std::fill(buffer_.begin(), buffer_.end(), 0.0);
        index_ = 0;
        size_ = 0;
        sum_ = 0.0;
    }

    ResultTuple call(const InputArray& inputs) override {
        const double high  = inputs[0];
        const double low   = inputs[1];
        const double close = inputs[2];
        const double tp = (high + low + close) / 3.0;

        // Rolling sum of TP via circular buffer.
        const double oldTP = buffer_[index_];
        buffer_[index_] = tp;
        index_++;
        if (index_ == window_size_) index_ = 0;
        if (size_ == window_size_) {
            sum_ += tp - oldTP;
        } else {
            sum_ += tp;
            size_++;
        }

        if (size_ < window_size_) {
            return std::numeric_limits<double>::quiet_NaN();
        }

        const double mean_tp = sum_ / window_size_;
        double mad = 0.0;
        for (int i = 0; i < window_size_; ++i) {
            mad += std::abs(buffer_[i] - mean_tp);
        }
        mad /= window_size_;
        if (mad == 0.0) {
            return 0.0;
        }
        return (tp - mean_tp) / (0.015 * mad);
    }

private:
    const int window_size_;
    std::vector<double> buffer_;
    int index_ = 0;
    int size_ = 0;
    double sum_ = 0.0;
};

}  // namespace screamer

#endif
