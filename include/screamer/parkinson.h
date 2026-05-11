#ifndef SCREAMER_PARKINSON_H
#define SCREAMER_PARKINSON_H

// Parkinson volatility (1980). Per-bar estimator using high and low:
//
//     sigma2_P[t] = (1 / (4 * ln 2)) * (ln(high/low))^2
//
// Averaged over n bars (rolling) or via an exponential weighting
// scheme to give an estimate of the variance of the log-price
// process. ~5x more efficient than close-to-close. Assumes zero
// drift and no overnight gaps.
//
// We expose two output forms:
//   RollingParkinsonVar / EwParkinsonVar  -- the variance estimate
//   RollingParkinsonVol / EwParkinsonVol  -- sqrt of the above
//
// All four are 2 -> 1 functors over (high, low). The *Vol classes
// hold a *Var member and apply sqrt; NaN propagates correctly.

#include <cmath>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include "screamer/common/functor_base.h"
#include "screamer/detail/rolling_mean.h"
#include "screamer/ew_mean.h"

namespace screamer {

namespace detail {

inline double parkinson_per_bar(double high, double low) {
    static constexpr double kInv4Ln2 = 1.0 / (4.0 * 0.6931471805599453);
    const double r = std::log(high / low);
    return kInv4Ln2 * r * r;
}

}  // namespace detail


class RollingParkinsonVar : public FunctorBase<RollingParkinsonVar, 2, 1> {
public:
    explicit RollingParkinsonVar(int window_size)
        : window_size_(window_size),
          mean_(static_cast<size_t>(window_size), "expanding")
    {
        if (window_size < 1) {
            throw std::invalid_argument("Window size must be positive.");
        }
    }

    void reset() override {
        mean_.reset();
        n_seen_ = 0;
    }

    ResultTuple call(const InputArray& inputs) override {
        const double per_bar = screamer::detail::parkinson_per_bar(inputs[0], inputs[1]);
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


class RollingParkinsonVol : public FunctorBase<RollingParkinsonVol, 2, 1> {
public:
    explicit RollingParkinsonVol(int window_size) : var_(window_size) {}

    void reset() override { var_.reset(); }

    ResultTuple call(const InputArray& inputs) override {
        const double v = var_.call(inputs);
        return std::sqrt(v);
    }

private:
    RollingParkinsonVar var_;
};


class EwParkinsonVar : public FunctorBase<EwParkinsonVar, 2, 1> {
public:
    explicit EwParkinsonVar(
        std::optional<double> com = std::nullopt,
        std::optional<double> span = std::nullopt,
        std::optional<double> halflife = std::nullopt,
        std::optional<double> alpha = std::nullopt)
        : mean_(com, span, halflife, alpha) {}

    void reset() override { mean_.reset(); }

    ResultTuple call(const InputArray& inputs) override {
        const double per_bar = screamer::detail::parkinson_per_bar(inputs[0], inputs[1]);
        return mean_.process_scalar(per_bar);
    }

private:
    EwMean mean_;
};


class EwParkinsonVol : public FunctorBase<EwParkinsonVol, 2, 1> {
public:
    explicit EwParkinsonVol(
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
    EwParkinsonVar var_;
};

}  // namespace screamer

#endif
