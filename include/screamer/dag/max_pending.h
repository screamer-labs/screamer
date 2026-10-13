#ifndef SCREAMER_DAG_MAX_PENDING_H
#define SCREAMER_DAG_MAX_PENDING_H

// max_pending: the most events a join may hold while it waits for another
// stream (the reorder buffer of CombineLatest, Filter and OrderedMerge nodes,
// and the rows of a Pipeline's output alignment that wait for another output
// in a live session). One default and one rule for every
// binding.

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>

namespace screamer { namespace dag {

inline constexpr std::size_t kDefaultMaxPending = 1'000'000;

// The largest max_pending: 2**53, the largest integer every binding holds
// exactly (a JavaScript number).
inline constexpr double kMaxMaxPending = 9007199254740992.0;

// A setting of the wrong kind: TypeError in Python and JavaScript.
struct SettingTypeError : std::invalid_argument {
    using std::invalid_argument::invalid_argument;
};

// max_pending that is not a number; `kind` names its type ("str", "a string").
// `name` is the binding's name of the setting (max_pending, maxPending).
[[noreturn]] inline void throw_max_pending_kind(const std::string& kind,
                                                const std::string& name = "max_pending") {
    throw SettingTypeError(name + " must be an integer, got " + kind);
}

// The shortest text that reads back as `v` ("1.5", "1e+300", "NaN").
inline std::string number_text(double v) {
    if (std::isnan(v)) return "NaN";
    char buf[32];
    for (int prec = 1; prec <= 17; ++prec) {
        std::snprintf(buf, sizeof buf, "%.*g", prec, v);
        if (!std::isfinite(v) || std::strtod(buf, nullptr) == v) break;
    }
    return buf;
}

// A count setting (max_pending, max_fill) from a number: a whole number from
// 1 to 2**53. A fractional or non-finite number is a TypeError; a whole number
// out of range a ValueError. `name` is the binding's name of the setting.
inline double checked_count_setting(double v, const std::string& name) {
    if (!std::isfinite(v) || v != std::floor(v)) throw_max_pending_kind(number_text(v), name);
    if (v < 1 || v > kMaxMaxPending)
        throw std::invalid_argument(
            name + " must be at least 1 and at most 2**53, got " + number_text(v));
    return v;
}

// A fill Resample's max_fill (resample_params.h kDefaultMaxFill).
inline std::int64_t checked_max_fill(double v, const std::string& name = "max_fill") {
    return static_cast<std::int64_t>(checked_count_setting(v, name));
}

// max_pending from a number, under the count-setting rule.
inline std::size_t checked_max_pending(double v, const std::string& name = "max_pending") {
    checked_count_setting(v, name);
    // A cap beyond what size_t holds (a 32-bit WebAssembly build) holds no
    // fewer events than memory can.
    if (v >= static_cast<double>(SIZE_MAX)) return SIZE_MAX;
    return static_cast<std::size_t>(v);
}

}}  // namespace screamer::dag
#endif
