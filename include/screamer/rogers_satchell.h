#ifndef SCREAMER_ROGERS_SATCHELL_H
#define SCREAMER_ROGERS_SATCHELL_H

// Rogers-Satchell volatility (1991). Per-bar estimator using OHLC:
//
//     sigma2_RS[t] = ln(H/C) * ln(H/O) + ln(L/C) * ln(L/O)
//
// Averaged over n bars (rolling) or EW. ~6x more efficient than
// close-to-close and -- crucially -- DRIFT-ROBUST: unlike Parkinson
// and Garman-Klass it does not assume the underlying process has
// zero drift, so it handles trending markets correctly.
//
// 4 -> 1 over (open, high, low, close). Still assumes no overnight
// gaps (Yang-Zhang's role).

#include <cmath>
#include <limits>
#include <optional>
#include <stdexcept>
#include "screamer/common/functor_base.h"
#include "screamer/detail/rolling_mean.h"
#include "screamer/ew_mean.h"

namespace screamer {

namespace detail {

inline double rogers_satchell_per_bar(double open, double high, double low, double close) {
    return std::log(high / close) * std::log(high / open)
         + std::log(low  / close) * std::log(low  / open);
}

}  // namespace detail


class RollingRogersSatchellVar : public FunctorBase<RollingRogersSatchellVar, 4, 1> {
public:
    explicit RollingRogersSatchellVar(int window_size)
        : window_size_(window_size),
          mean_(static_cast<size_t>(window_size), "expanding")
    {
        if (window_size < 1) {
            throw std::invalid_argument("Window size must be positive.");
        }
    }
    void reset() override { mean_.reset(); n_seen_ = 0; }
    ResultTuple call(const InputArray& inputs) override {
        const double per_bar = screamer::detail::rogers_satchell_per_bar(
            inputs[0], inputs[1], inputs[2], inputs[3]);
        const double v = mean_.append(per_bar);
        if (n_seen_ < window_size_) {
            n_seen_++;
            if (n_seen_ < window_size_) {
                return std::numeric_limits<double>::quiet_NaN();
            }
        }
        return v;
    }
private:
    const int window_size_;
    screamer::detail::RollingMean mean_;
    int n_seen_ = 0;
};


class RollingRogersSatchellVol : public FunctorBase<RollingRogersSatchellVol, 4, 1> {
public:
    explicit RollingRogersSatchellVol(int window_size) : var_(window_size) {}
    void reset() override { var_.reset(); }
    ResultTuple call(const InputArray& inputs) override {
        return std::sqrt(var_.call(inputs));
    }
private:
    RollingRogersSatchellVar var_;
};


class EwRogersSatchellVar : public FunctorBase<EwRogersSatchellVar, 4, 1> {
public:
    explicit EwRogersSatchellVar(
        std::optional<double> com = std::nullopt,
        std::optional<double> span = std::nullopt,
        std::optional<double> halflife = std::nullopt,
        std::optional<double> alpha = std::nullopt)
        : mean_(com, span, halflife, alpha) {}
    void reset() override { mean_.reset(); }
    ResultTuple call(const InputArray& inputs) override {
        const double per_bar = screamer::detail::rogers_satchell_per_bar(
            inputs[0], inputs[1], inputs[2], inputs[3]);
        return mean_.process_scalar(per_bar);
    }
private:
    EwMean mean_;
};


class EwRogersSatchellVol : public FunctorBase<EwRogersSatchellVol, 4, 1> {
public:
    explicit EwRogersSatchellVol(
        std::optional<double> com = std::nullopt,
        std::optional<double> span = std::nullopt,
        std::optional<double> halflife = std::nullopt,
        std::optional<double> alpha = std::nullopt)
        : var_(com, span, halflife, alpha) {}
    void reset() override { var_.reset(); }
    ResultTuple call(const InputArray& inputs) override {
        return std::sqrt(var_.call(inputs));
    }
private:
    EwRogersSatchellVar var_;
};

}  // namespace screamer

#endif
