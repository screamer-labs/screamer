#ifndef SCREAMER_COMMON_INT_SETTINGS_H
#define SCREAMER_COMMON_INT_SETTINGS_H

// The message for an integer setting below its smallest value, one form for
// every operator and both bindings: "<Op>: <setting> must be at least 1, got
// -1". A window, period or count below it would otherwise fail inside the
// operator with whatever its buffers raise (such as std::length_error's
// "vector" for a negative size).
//
// As with finite_settings.h, each operator's constructor refuses a value it
// cannot take, and the binding asks int_setting_error() on the error path
// only whether one of the integer settings it was given is below the
// smallest value the operator ever takes (over its other settings, string
// settings such as fill_at included); then this message replaces the
// constructor's. kIntSettingMin holds that smallest value for every integer
// setting (the smallest any combination of the other settings allows; a
// value at or above it that the constructor still refuses, such as a MACD
// slow span below fast, keeps the constructor's message).
// tests/test_int_settings.py checks the table against the constructors.

#include <cstddef>
#include <limits>
#include <string>
#include <vector>
#include "screamer/common/finite_settings.h"

namespace screamer {

struct IntSettingMin {
    const char* op;
    const char* setting;     // the Python (snake_case) name
    long long min;
};

inline constexpr IntSettingMin kIntSettingMin[] = {
    {"ADOSC", "fast", 2},
    {"ADOSC", "slow", 3},
    {"ADX", "window_size", 2},
    {"ADXR", "window_size", 2},
    {"ATR", "window_size", 2},
    {"AmihudIlliquidity", "window_size", 1},
    {"BacktestOHLCTarget", "delay", 0},
    {"BacktestOHLCVTarget", "delay", 0},
    {"BacktestPortfolioOHLCTarget", "delay", 0},
    {"BollingerBands", "window_size", 2},
    {"BulkVolumeClassifier", "window_size", 2},
    {"Butter", "order", 1},
    {"ButterBandpass", "order", 1},
    {"ButterBandstop", "order", 1},
    {"ButterHighpass", "order", 1},
    {"CCI", "window_size", 2},
    {"CrossTopBottom", "count", 1},
    {"DX", "window_size", 2},
    {"Detrend", "window_size", 1},
    {"Diff", "window_size", 1},
    {"DonchianChannels", "window_size", 1},
    {"FracDiff", "window_size", 1},
    {"Hampel", "window_size", 1},
    {"Hold", "cooldown", 0},
    {"Hold", "n", 1},
    {"HullMA", "window_size", 4},
    {"ImpulseClip", "window_size", 1},
    {"KAMA", "fast", 1},
    {"KAMA", "slow", 1},
    {"KAMA", "window_size", 2},
    {"KeltnerChannels", "window_size", 2},
    {"Lag", "window_size", 1},
    {"LogReturn", "window_size", 1},
    {"MACD", "fast", 2},
    {"MACD", "signal", 2},
    {"MACD", "slow", 3},
    {"MFI", "window_size", 1},
    {"MinusDI", "window_size", 2},
    {"MinusDM", "window_size", 2},
    {"Momentum", "window_size", 1},
    {"NATR", "window_size", 2},
    {"PlusDI", "window_size", 2},
    {"PlusDM", "window_size", 2},
    {"Propagator", "window_size", 2},
    {"ROC", "window_size", 1},
    {"ROCP", "window_size", 1},
    {"ROCR", "window_size", 1},
    {"RealizedSpread", "lag", 1},
    {"Return", "window_size", 1},
    {"RollSpread", "window_size", 2},
    {"RollingADF", "lags", 0},
    {"RollingADF", "window_size", 4},
    {"RollingAlpha", "window_size", 2},
    {"RollingArgmax", "window_size", 1},
    {"RollingArgmin", "window_size", 1},
    {"RollingBeta", "window_size", 2},
    {"RollingCVaR", "window_size", 1},
    {"RollingCalmar", "window_size", 2},
    {"RollingBipowerVar", "window_size", 2},
    {"RollingCorr", "window_size", 2},
    {"RollingCov", "window_size", 2},
    {"RollingDownsideDeviation", "window_size", 2},
    {"RollingGarmanKlassVar", "window_size", 1},
    {"RollingGarmanKlassVol", "window_size", 1},
    {"RollingHitRate", "window_size", 1},
    {"RollingHurst", "min_scale", 4},
    {"RollingHurst", "window_size", 32},
    {"RollingInfoRatio", "window_size", 2},
    {"RollingIqr", "window_size", 1},
    {"RollingJumpShare", "window_size", 2},
    {"RollingKurt", "window_size", 1},
    {"RollingKyleLambda", "window_size", 2},
    {"RollingLinearRegression", "window_size", 3},
    {"RollingMad", "window_size", 1},
    {"RollingMax", "window_size", 1},
    {"RollingMaxDrawdown", "window_size", 2},
    {"RollingMean", "window_size", 1},
    {"RollingMedian", "window_size", 1},
    {"RollingMedianAD", "window_size", 1},
    {"RollingMin", "window_size", 1},
    {"RollingMinMax", "window_size", 1},
    {"RollingOU", "window_size", 3},
    {"RollingOmega", "window_size", 2},
    {"RollingOrderImbalance", "window_size", 1},
    {"RollingParkinsonVar", "window_size", 1},
    {"RollingParkinsonVol", "window_size", 1},
    {"RollingPercentile", "window_size", 1},
    {"RollingPoly1", "derivative_order", 0},
    {"RollingPoly1", "window_size", 2},
    {"RollingPoly2", "derivative_order", 0},
    {"RollingPoly2", "window_size", 3},
    {"RollingQuantile", "window_size", 1},
    {"RollingRSI", "window_size", 2},
    {"RollingRange", "window_size", 1},
    {"RollingRank", "window_size", 1},
    {"RollingResidualStd", "window_size", 2},
    {"RollingRms", "window_size", 2},
    {"RollingRogersSatchellVar", "window_size", 1},
    {"RollingRogersSatchellVol", "window_size", 1},
    {"RollingSeasonalMean", "period", 1},
    {"RollingSeasonalMean", "window_size", 1},
    {"RollingSeasonalStd", "period", 1},
    {"RollingSeasonalStd", "window_size", 2},
    {"RollingSeasonalZscore", "period", 1},
    {"RollingSeasonalZscore", "window_size", 2},
    {"RollingSharpe", "window_size", 2},
    {"RollingSigmaClip", "window_size", 1},
    {"RollingSkew", "window_size", 3},
    {"RollingSortino", "window_size", 2},
    {"RollingSpread", "window_size", 2},
    {"RollingStd", "window_size", 2},
    {"RollingSum", "window_size", 1},
    {"RollingTLSBeta", "window_size", 2},
    {"RollingTSF", "window_size", 2},
    {"RollingUpsideDeviation", "window_size", 2},
    {"RollingVarianceRatio", "lag", 2},
    {"RollingVarianceRatio", "window_size", 5},
    {"RollingVWAP", "window_size", 1},
    {"RollingVar", "window_size", 2},
    {"RollingYangZhangVar", "window_size", 2},
    {"RollingYangZhangVol", "window_size", 2},
    {"RollingZscore", "window_size", 2},
    {"Stoch", "d", 1},
    {"Stoch", "fastk_period", 1},
    {"Stoch", "smooth_k", 1},
    {"StochRSI", "d", 1},
    {"StochRSI", "rsi_period", 2},
    {"StochRSI", "smooth_k", 1},
    {"StochRSI", "stoch_period", 1},
    {"TRIMA", "window_size", 1},
    {"TRIX", "span", 2},
    {"TripleBarrier", "horizon", 1},
    {"TripleBarrierOHLC", "horizon", 1},
    {"UltimateOscillator", "period1", 1},
    {"UltimateOscillator", "period2", 1},
    {"UltimateOscillator", "period3", 1},
    {"VPIN", "n_buckets", 1},
    {"WMA", "window_size", 1},
    {"WilliamsR", "window_size", 1},
};

// The smallest value of an integer setting, or false when the table has no entry.
inline bool int_setting_min(const std::string& op, const std::string& setting, long long& min) {
    const std::string key = detail::setting_key(setting);
    for (const auto& m : kIntSettingMin)
        if (op == m.op && key == detail::setting_key(m.setting)) { min = m.min; return true; }
    return false;
}

// The largest value of an integer setting: the range of the C++ int every
// integer setting is held in. A larger value is refused before it is
// narrowed, in both bindings, never wrapped.
inline constexpr long long kIntSettingMax = std::numeric_limits<int>::max();

// The message for the first integer setting, in the order given, outside
// its range: "<op>: <setting> must be at least <min>, got <text>" below its
// smallest value, "<op>: <setting> must be at most 2147483647, got <text>"
// above the int range. `texts` holds each value as the caller wrote it
// (Python's int, JavaScript's String(x)), so the message shows the value
// given. With `alloc_failed` (the constructor ran out of memory, or a
// buffer's std::length_error), and no value out of range, the message names
// the largest setting: "<op>: <setting> of <text> needs more memory than is
// available". "" when there is none.
inline std::string int_setting_error(const std::string& op, const std::vector<std::string>& names,
                                     const std::vector<double>& values,
                                     const std::vector<std::string>& texts,
                                     bool alloc_failed = false) {
    const auto text = [&](std::size_t i) {
        return i < texts.size() ? texts[i] : std::string("?");
    };
    std::ptrdiff_t largest = -1;
    for (std::size_t i = 0; i < names.size() && i < values.size(); ++i) {
        long long min = 0;
        if (!int_setting_min(op, names[i], min)) continue;
        if (values[i] < static_cast<double>(min))
            return op + ": " + names[i] + " must be at least " + std::to_string(min) + ", got " + text(i);
        if (values[i] > static_cast<double>(kIntSettingMax))
            return op + ": " + names[i] + " must be at most " + std::to_string(kIntSettingMax) +
                   ", got " + text(i);
        if (largest < 0 || values[i] > values[static_cast<std::size_t>(largest)])
            largest = static_cast<std::ptrdiff_t>(i);
    }
    if (alloc_failed && largest >= 0)
        return op + ": " + names[static_cast<std::size_t>(largest)] + " of " +
               text(static_cast<std::size_t>(largest)) + " needs more memory than is available";
    return std::string();
}

}  // namespace screamer
#endif
