#ifndef SCREAMER_GARMAN_KLASS_H
#define SCREAMER_GARMAN_KLASS_H

// Garman-Klass volatility (1980). Per-bar estimator using OHLC:
//
//     sigma2_GK[t] = 0.5 * (ln(H/L))^2 - (2*ln 2 - 1) * (ln(C/O))^2
//
// Averaged over n bars (rolling) or EW. ~7.4x more efficient than
// close-to-close. Assumes zero drift and no overnight gaps.
//
// 4 -> 1 over (open, high, low, close). Argument order matches the
// general OHLC convention used elsewhere in the library (BOP, etc.).

#include <cmath>
#include <limits>
#include <optional>
#include <stdexcept>
#include "screamer/common/functor_base.h"
#include "screamer/detail/rolling_mean.h"
#include "screamer/ew_mean.h"

namespace screamer {

namespace detail {

inline double garman_klass_per_bar(double open, double high, double low, double close) {
    static constexpr double k2Ln2Minus1 = 2.0 * 0.6931471805599453 - 1.0;
    const double rhl = std::log(high / low);
    const double rco = std::log(close / open);
    return 0.5 * rhl * rhl - k2Ln2Minus1 * rco * rco;
}

}  // namespace detail


class RollingGarmanKlassVar : public FunctorBase<RollingGarmanKlassVar, 4, 1> {
public:
    explicit RollingGarmanKlassVar(int window_size)
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
        const double per_bar = screamer::detail::garman_klass_per_bar(
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


class RollingGarmanKlassVol : public FunctorBase<RollingGarmanKlassVol, 4, 1> {
public:
    explicit RollingGarmanKlassVol(int window_size) : var_(window_size) {}
    void reset() override { var_.reset(); }
    ResultTuple call(const InputArray& inputs) override {
        return std::sqrt(var_.call(inputs));
    }
private:
    RollingGarmanKlassVar var_;
};


class EwGarmanKlassVar : public FunctorBase<EwGarmanKlassVar, 4, 1> {
public:
    explicit EwGarmanKlassVar(
        std::optional<double> com = std::nullopt,
        std::optional<double> span = std::nullopt,
        std::optional<double> halflife = std::nullopt,
        std::optional<double> alpha = std::nullopt)
        : mean_(com, span, halflife, alpha) {}
    void reset() override { mean_.reset(); }
    ResultTuple call(const InputArray& inputs) override {
        const double per_bar = screamer::detail::garman_klass_per_bar(
            inputs[0], inputs[1], inputs[2], inputs[3]);
        return mean_.process_scalar(per_bar);
    }
private:
    EwMean mean_;
};


class EwGarmanKlassVol : public FunctorBase<EwGarmanKlassVol, 4, 1> {
public:
    explicit EwGarmanKlassVol(
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
    EwGarmanKlassVar var_;
};

}  // namespace screamer

#endif
