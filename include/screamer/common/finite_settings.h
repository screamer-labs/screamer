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
#include <stdexcept>
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
    // Position bounds: an infinite bound is no bound (the default).
    {"BacktestL1Orders", "max_position", kTakesPosInf},
    {"BacktestL1Orders", "min_position", kTakesNegInf},
    {"BacktestL1Target", "max_position", kTakesPosInf},
    {"BacktestL1Target", "min_position", kTakesNegInf},
    {"BacktestL1TradesOrders", "max_position", kTakesPosInf},
    {"BacktestL1TradesOrders", "min_position", kTakesNegInf},
    {"BacktestOHLCOrders", "max_position", kTakesPosInf},
    {"BacktestOHLCOrders", "min_position", kTakesNegInf},
    {"BacktestOHLCTarget", "max_position", kTakesPosInf},
    {"BacktestOHLCTarget", "min_position", kTakesNegInf},
    {"BacktestOHLCVTarget", "max_position", kTakesPosInf},
    {"BacktestOHLCVTarget", "min_position", kTakesNegInf},
    {"BacktestPriceTarget", "max_position", kTakesPosInf},
    {"BacktestPriceTarget", "min_position", kTakesNegInf},
    {"BacktestTradesOrders", "max_position", kTakesPosInf},
    {"BacktestTradesOrders", "min_position", kTakesNegInf},
    {"BacktestTradesTarget", "max_position", kTakesPosInf},
    {"BacktestTradesTarget", "min_position", kTakesNegInf},
    // Exposure and participation caps: inf is no cap (the default).
    {"BacktestPortfolioOHLCTarget", "max_gross", kTakesPosInf},
    {"BacktestPortfolioOHLCTarget", "max_net", kTakesPosInf},
    {"BacktestPortfolioTarget", "max_gross", kTakesPosInf},
    {"BacktestPortfolioTarget", "max_net", kTakesPosInf},
    {"BacktestTradesTarget", "max_participation", kTakesPosInf},
    // inf is no stop (the default).
    {"BandSignal", "stop", kTakesPosInf},
    // Bounds: NaN is None (no bound), and so is an infinity on the bound's
    // own side.
    {"Clip", "lower", kTakesNegInf | kTakesNaN},
    {"Clip", "upper", kTakesPosInf | kTakesNaN},
    {"RollingSigmaClip", "lower", kTakesNegInf | kTakesNaN},
    {"RollingSigmaClip", "upper", kTakesPosInf | kTakesNaN},
    // NaN is None: no net target.
    {"CrossNormalize", "net", kTakesNaN},
    // The value that replaces NaN: any float (NaN leaves NaN in place).
    {"FillNa", "fill", kTakesNonFinite},
    // NaN is a documented output or state.
    {"Hold", "release", kTakesNaN},
    {"SchmittTrigger", "initial", kTakesNaN},
};

// The check an operator's constructor makes on a float setting: a value that
// is not finite raises std::invalid_argument unless `takes` (kTakes* bits)
// allows it. The bindings replace the message with finite_setting_error's,
// so `takes` must match the setting's kNonFiniteAllowed entry.
inline void require_finite(const char* setting, double v, unsigned takes = 0u) {
    if (std::isfinite(v)) return;
    const unsigned bit = std::isnan(v) ? kTakesNaN : v > 0 ? kTakesPosInf : kTakesNegInf;
    if (!(takes & bit)) throw std::invalid_argument(std::string(setting) + " must be finite");
}

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
