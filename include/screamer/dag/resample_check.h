#ifndef SCREAMER_DAG_RESAMPLE_CHECK_H
#define SCREAMER_DAG_RESAMPLE_CHECK_H
// The settings rule of a Resample, shared by the Python and the JavaScript
// bindings: what a call gave (ResampleSettings) is checked in one place, with
// one set of messages. A value error is std::invalid_argument (ValueError in
// Python, RangeError in JavaScript); a kind error is SettingTypeError
// (TypeError). Each binding first refuses a setting of the wrong kind (a
// string or a boolean) with its own type names, then calls this.

#include <cmath>
#include <stdexcept>
#include <string>
#include "screamer/dag/max_pending.h"

namespace screamer { namespace dag {

struct ResampleSettings {
    bool has_freq = false;                 // Python's freq= (a time span)
    bool has_every = false;     double every = 0;
    bool has_count = false;     double count = 0;
    bool has_threshold = false; double threshold = 0;
    bool clock = false;
    bool agg_functor = false;              // a functor reducer, not a name
    std::string agg = "last";              // the name, aliases resolved
    std::string label = "left";
    std::string fill = "skip";
    bool has_origin = false;    double origin = 0;
    bool has_max_age = false;
    bool max_age_whole = true;             // the binding read an integer
    double max_age = 0;
};

namespace detail {
inline bool resample_agg_known(const std::string& a) {
    static const char* const names[] = {"first", "last", "min", "max", "sum", "count",
                                        "mean", "ohlc", "ohlcv", "ohlcv2", "ohlc_bars",
                                        "ohlcv_bars"};
    for (const char* n : names) if (a == n) return true;
    return false;
}
// An integer setting past the int64 range (inf included).
inline void check_int64_setting(double v, const char* name) {
    if (std::fabs(v) >= 9223372036854775808.0)
        throw std::invalid_argument(std::string("resample: ") + name +
                                    " must be within the int64 range, got " + number_text(v));
}
}  // namespace detail

// Raises the first error of `s`, in the order the Python Resample checked
// them before 3.0.0.
inline void check_resample_settings(const ResampleSettings& s) {
    if (s.has_max_age) {
        if (!s.clock || s.agg_functor || s.agg != "last" || s.fill != "carry")
            throw std::invalid_argument(
                "resample: max_age requires clock=True, agg='last', and fill='carry'");
        if (!s.max_age_whole)
            throw SettingTypeError("resample: max_age must be a non-negative integer in index units");
        if (!(s.max_age >= 0))
            throw std::invalid_argument("resample: max_age must be a non-negative integer");
        detail::check_int64_setting(s.max_age, "max_age");
    }
    const int given = int(s.has_freq) + int(s.has_every) + int(s.has_count) +
                      int(s.has_threshold) + int(s.clock);
    if (given != 1)
        throw std::invalid_argument(
            "resample: pass exactly one of freq=, every=, count=, threshold=, or clock=");
    if (s.has_threshold && !(s.threshold > 0.0))
        throw std::invalid_argument("resample: threshold must be > 0");
    if (s.has_threshold && !s.agg_functor &&
        (s.agg == "ohlcv" || s.agg == "ohlcv2" || s.agg == "ohlc_bars" || s.agg == "ohlcv_bars"))
        throw std::invalid_argument(
            "resample threshold= mode supports single-column value aggs "
            "(first/last/min/max/sum/count/mean/ohlc); agg='" + s.agg + "' requires "
            "multi-column value input that is not available in threshold= mode. "
            "For OHLCV volume bars, resample the OHLC and the volume separately and "
            "combine: e.g. combine_latest(Resample(price_vi, vol, threshold=T, agg='ohlc'), "
            "Resample(price_vi, vol, threshold=T, agg='sum')).");
    // every=0 would divide by zero in the engine; a count below 1 never
    // completes a bar. A fraction is truncated, as int() does.
    if (s.has_every) {
        if (!(s.every >= 1)) throw std::invalid_argument("resample: every must be positive");
        detail::check_int64_setting(s.every, "every");
    }
    if (s.has_count) {
        if (!(s.count >= 1)) throw std::invalid_argument("resample: count must be >= 1");
        detail::check_int64_setting(s.count, "count");
    }
    if (s.has_origin) {
        if (std::isnan(s.origin))
            throw std::invalid_argument("resample: origin must be an integer, got nan");
        detail::check_int64_setting(s.origin, "origin");
    }
    // A functor reducer runs on the generic node, which has no clock or
    // cumulative mode: it would give silently wrong bars.
    if (s.agg_functor && s.clock)
        throw std::invalid_argument(
            "resample clock= mode requires a builtin string agg "
            "(first/last/min/max/sum/count/mean/ohlc); "
            "functor reducers are not supported with clock=. "
            "Apply your functor to the resampled output instead.");
    if (s.agg_functor && s.has_threshold)
        throw std::invalid_argument(
            "resample threshold= mode requires a builtin string agg "
            "(first/last/min/max/sum/count/mean/ohlc); "
            "functor reducers are not supported with threshold=. "
            "Apply your functor to the resampled output instead.");
    if (!s.agg_functor && !detail::resample_agg_known(s.agg))
        throw std::invalid_argument(
            "resample: unknown agg string '" + s.agg + "'; valid names are "
            "['close', 'count', 'first', 'high', 'kurt', 'last', 'low', 'max', 'mean', "
            "'min', 'ohlc', 'ohlc_bars', 'ohlcv', 'ohlcv2', 'ohlcv_bars', 'open', 'prod', "
            "'skew', 'std', 'sum', 'var']");
    if (s.label != "left" && s.label != "right")
        throw std::invalid_argument("resample: label must be \"left\" or \"right\"");
    if (s.fill != "skip" && s.fill != "nan" && s.fill != "carry")
        throw std::invalid_argument("resample: fill must be one of ('skip', 'nan', 'carry')");
}

// The engine's own guard (GraphBuilder::add_resample): settings that would
// divide by zero or never close a bar, for a description that did not come
// through a Resample factory.
inline void check_resample_params_numbers(int mode, std::int64_t width, std::int64_t count,
                                          double threshold) {
    if (mode == 0 && width <= 0) throw std::invalid_argument("resample: every must be positive");
    if (mode == 1 && count < 1) throw std::invalid_argument("resample: count must be >= 1");
    if (mode == 2 && !(threshold > 0.0))
        throw std::invalid_argument("resample: threshold must be > 0");
}

}}  // namespace screamer::dag
#endif
