#ifndef SCREAMER_COMMON_FINITE_SETTINGS_H
#define SCREAMER_COMMON_FINITE_SETTINGS_H

// The message for an operator setting that is not finite, one form for every
// operator and both bindings: "<Op>: <setting> must be finite, got inf", or
// "must be finite or NaN, got inf" for a setting that also takes NaN.
//
// Each operator's constructor checks its own settings and refuses a value it
// cannot take. When a constructor refuses, the binding asks
// finite_setting_error() whether one of the float settings it was given is
// not finite where the operator needs a finite number, and if so raises this
// message in place of the constructor's (ValueError in Python, RangeError in
// JavaScript). It runs on the error path only, so a construction that succeeds
// costs nothing. The setting is named as the caller passed it (num_std in
// Python, numStd in JavaScript).
//
// Every float setting must be finite, except the ones listed in
// kNonFiniteAllowed with the non-finite values they take (an infinite cap
// meaning "no cap", NaN meaning "none"). tests/test_setting_kinds.py checks
// the list against the constructors.

#include <cmath>
#include <cstddef>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

namespace screamer {

enum : unsigned { kTakesPosInf = 1u, kTakesNegInf = 2u, kTakesNaN = 4u,
                  kTakesNonFinite = 7u };

struct NonFiniteAllowed {
    const char* op;
    const char* setting;     // the Python (snake_case) name
    unsigned takes;          // kTakes* bits
};

inline constexpr NonFiniteAllowed kNonFiniteAllowed[] = {
    // Backtest engines: infinite position limits, fees and caps.
    {"BacktestL1Orders", "maker_fee", kTakesNonFinite},
    {"BacktestL1Orders", "taker_fee", kTakesNonFinite},
    {"BacktestL1Orders", "max_position", kTakesNonFinite},
    {"BacktestL1Orders", "min_position", kTakesNonFinite},
    {"BacktestL1Target", "taker_fee", kTakesNonFinite},
    {"BacktestL1Target", "max_position", kTakesNonFinite},
    {"BacktestL1Target", "min_position", kTakesNonFinite},
    {"BacktestL1TradesOrders", "maker_fee", kTakesNonFinite},
    {"BacktestL1TradesOrders", "taker_fee", kTakesNonFinite},
    {"BacktestL1TradesOrders", "maker_fee_per_contract", kTakesNonFinite},
    {"BacktestL1TradesOrders", "taker_fee_per_contract", kTakesNonFinite},
    {"BacktestL1TradesOrders", "max_position", kTakesNonFinite},
    {"BacktestL1TradesOrders", "min_position", kTakesNonFinite},
    {"BacktestOHLCOrders", "maker_fee", kTakesNonFinite},
    {"BacktestOHLCOrders", "taker_fee", kTakesNonFinite},
    {"BacktestOHLCOrders", "max_position", kTakesNonFinite},
    {"BacktestOHLCOrders", "min_position", kTakesNonFinite},
    {"BacktestOHLCTarget", "taker_fee", kTakesNonFinite},
    {"BacktestOHLCTarget", "max_position", kTakesNonFinite},
    {"BacktestOHLCTarget", "min_position", kTakesNonFinite},
    {"BacktestOHLCVTarget", "taker_fee", kTakesNonFinite},
    {"BacktestOHLCVTarget", "max_position", kTakesNonFinite},
    {"BacktestOHLCVTarget", "min_position", kTakesNonFinite},
    {"BacktestPortfolioOHLCTarget", "max_gross", kTakesPosInf},
    {"BacktestPortfolioOHLCTarget", "max_net", kTakesPosInf},
    {"BacktestPortfolioTarget", "max_gross", kTakesPosInf},
    {"BacktestPortfolioTarget", "max_net", kTakesPosInf},
    {"BacktestPriceTarget", "fee", kTakesNonFinite},
    {"BacktestPriceTarget", "max_position", kTakesNonFinite},
    {"BacktestPriceTarget", "min_position", kTakesNonFinite},
    {"BacktestPriceTarget", "spread", kTakesPosInf | kTakesNaN},
    {"BacktestTradesOrders", "maker_fee", kTakesNonFinite},
    {"BacktestTradesOrders", "taker_fee", kTakesNonFinite},
    {"BacktestTradesOrders", "max_position", kTakesNonFinite},
    {"BacktestTradesOrders", "min_position", kTakesNonFinite},
    {"BacktestTradesTarget", "taker_fee", kTakesNonFinite},
    {"BacktestTradesTarget", "max_position", kTakesNonFinite},
    {"BacktestTradesTarget", "min_position", kTakesNonFinite},
    {"BacktestTradesTarget", "max_participation", kTakesPosInf},
    {"BandSignal", "stop", kTakesPosInf},
    {"BayesianRegression", "prior_precision", kTakesPosInf | kTakesNaN},
    {"BayesianRegression", "prior_sigma", kTakesPosInf | kTakesNaN},
    {"Butter", "cutoff_freq", kTakesNonFinite},
    {"Clip", "lower", kTakesNonFinite},
    {"Clip", "upper", kTakesNonFinite},
    {"CrossNormalize", "net", kTakesNaN},
    {"Decycler", "period", kTakesPosInf},
    {"FillNa", "fill", kTakesNonFinite},
    {"FracDiff", "d", kTakesPosInf | kTakesNegInf},
    {"FracDiff", "threshold", kTakesPosInf},
    {"Hampel", "n_sigma", kTakesPosInf | kTakesNaN},
    {"HawkesIntensity", "alpha", kTakesNonFinite},
    {"HawkesIntensity", "mu", kTakesNonFinite},
    {"Hold", "release", kTakesNaN},
    {"ImpulseClip", "n_sigma", kTakesPosInf | kTakesNaN},
    {"KalmanFilter", "initial_state", kTakesNonFinite},
    {"KalmanFilter", "initial_variance", kTakesPosInf},
    {"KalmanFilter", "observation_var", kTakesPosInf},
    {"KalmanFilter", "process_var", kTakesPosInf},
    {"KeltnerChannels", "num_atr", kTakesNonFinite},
    {"Linear", "scale", kTakesNonFinite},
    {"Linear", "shift", kTakesNonFinite},
    {"Linear2", "a", kTakesNonFinite},
    {"Linear2", "b", kTakesNonFinite},
    {"Linear2", "c", kTakesNonFinite},
    {"MAMA", "fast_limit", kTakesNonFinite},
    {"MAMA", "slow_limit", kTakesNonFinite},
    {"MovingAverage", "taps", kTakesNonFinite},
    {"Power", "p", kTakesNonFinite},
    {"Propagator", "g0", kTakesNonFinite},
    {"Propagator", "gamma", kTakesNonFinite},
    {"RollingCVaR", "alpha", kTakesNaN},
    {"RollingCalmar", "periods_per_year", kTakesPosInf},
    {"RollingDownsideDeviation", "mar", kTakesNonFinite},
    {"RollingInfoRatio", "periods_per_year", kTakesPosInf},
    {"RollingOmega", "threshold", kTakesNonFinite},
    {"RollingQuantile", "quantile", kTakesNaN},
    {"RollingSharpe", "periods_per_year", kTakesPosInf},
    {"RollingSigmaClip", "lower", kTakesNegInf | kTakesNaN},
    {"RollingSigmaClip", "upper", kTakesPosInf | kTakesNaN},
    {"RollingSortino", "periods_per_year", kTakesPosInf},
    {"RollingSortino", "target", kTakesNonFinite},
    {"RoofingFilter", "hp_period", kTakesPosInf},
    {"RoofingFilter", "lp_period", kTakesPosInf},
    {"SchmittTrigger", "initial", kTakesNaN},
    {"SuperSmoother", "period", kTakesPosInf},
    {"VPIN", "bucket_volume", kTakesPosInf | kTakesNaN},
};

namespace detail {

// A setting name in one form for both bindings: lower case letters and
// digits, so num_std and numStd compare equal.
inline std::string setting_key(const std::string& name) {
    std::string k;
    for (char c : name) {
        if (c == '_') continue;
        k += (c >= 'A' && c <= 'Z') ? static_cast<char>(c - 'A' + 'a') : c;
    }
    return k;
}

inline unsigned non_finite_taken(const std::string& op, const std::string& setting) {
    const std::string key = setting_key(setting);
    for (const auto& a : kNonFiniteAllowed)
        if (op == a.op && key == setting_key(a.setting)) return a.takes;
    return 0u;
}

// inf, -inf and nan in Python's spelling, or JavaScript's.
inline std::string non_finite_text(double v, bool js) {
    if (std::isnan(v)) return js ? "NaN" : "nan";
    return v > 0 ? (js ? "Infinity" : "inf") : (js ? "-Infinity" : "-inf");
}

}  // namespace detail

// "<op>: <setting> must be finite, got inf" for the first setting, in the
// order given, that holds a value the operator takes only as a finite number;
// "" when there is none. values[i] holds setting i's numbers (one, or one per
// asset or tap); `js` spells the value as JavaScript does.
inline std::string finite_setting_error(const std::string& op,
                                        const std::vector<std::string>& names,
                                        const std::vector<std::vector<double>>& values,
                                        bool js) {
    for (std::size_t i = 0; i < names.size() && i < values.size(); ++i) {
        const unsigned takes = detail::non_finite_taken(op, names[i]);
        for (double v : values[i]) {
            if (std::isfinite(v)) continue;
            const unsigned bit = std::isnan(v) ? kTakesNaN : v > 0 ? kTakesPosInf : kTakesNegInf;
            if (takes & bit) continue;
            // The non-finite values the setting does take, as alternatives.
            std::string also;
            if (takes & kTakesPosInf) also += " or " + detail::non_finite_text(std::numeric_limits<double>::infinity(), js);
            if (takes & kTakesNegInf) also += " or " + detail::non_finite_text(-std::numeric_limits<double>::infinity(), js);
            if (takes & kTakesNaN) also += " or " + detail::non_finite_text(std::numeric_limits<double>::quiet_NaN(), js);
            return op + ": " + names[i] + " must be finite" + also + ", got " +
                   detail::non_finite_text(v, js);
        }
    }
    return std::string();
}

}  // namespace screamer
#endif
