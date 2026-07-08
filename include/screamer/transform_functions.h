
#ifndef SCREAMER_TRANSFORM_FUNCTIONS_H
#define SCREAMER_TRANSFORM_FUNCTIONS_H

#include <cmath>

namespace screamer {

    // NaN policy "ignore": NaN propagates. The naive comparison-based
    // form returns 0 for NaN because both ``0 < NaN`` and ``NaN < 0`` are
    // false; the explicit guard below honors the policy.
    template <typename T> T signum(T val) {
        if (std::isnan(val)) return val;
        return (T(0) < val) - (val < T(0));
    }

    // Square and cube: precomposed common powers, faster than std::pow(x, 2)
    // and std::pow(x, 3) (no logarithm) and clearer than `Power(2)`.
    inline double square(double x) { return x * x; }
    inline double cube(double x)   { return x * x * x; }

    // Identity: pass-through. Useful as a placeholder node when a pipeline
    // slot needs filling without altering the data.
    inline double identity(double x) { return x; }

    // ReLU function. NaN propagates per the "ignore" NaN policy; the
    // naive form would return 0 because ``NaN > 0`` is false.
    inline double relu(double x) {
        if (std::isnan(x)) return x;
        return x > 0 ? x : 0;
    }

    // PosPart: positive part of x, i.e. max(x, 0). Identical to relu.
    // NaN propagates per the "ignore" policy.
    inline double pos_part(double x) { return std::isnan(x) ? x : (x > 0.0 ? x : 0.0); }

    // NegPart: negative part of x, i.e. max(-x, 0). Decomposes x = PosPart(x) - NegPart(x).
    // NaN propagates per the "ignore" policy.
    inline double neg_part(double x) { return std::isnan(x) ? x : (x < 0.0 ? -x : 0.0); }

    // ELU function (Exponential Linear Unit)
    inline double elu(double x) {
        return x > 0 ? x : (std::exp(x) - 1);
    }

    // SELU function (Scaled Exponential Linear Unit)
    inline double selu(double x) {
        const double lambda = 1.0507; // Scaling parameter
        const double alpha = 1.67326; // Exponential parameter
        return x > 0 ? lambda * x : lambda * alpha * (std::exp(x) - 1);
    }

    // Softsign function
    inline double softsign(double x) {
        return x / (1 + std::abs(x));
    }

    // Sigmoid function
    inline double sigmoid(double x) {
        return 1 / (1 + std::exp(-x));
    }

} // namespace

#endif