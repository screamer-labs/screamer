#ifndef SCREAMER_DAG_CHECKED_INDEX_H
#define SCREAMER_DAG_CHECKED_INDEX_H

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace screamer { namespace dag {

// The kind error for an index the int64-indexed engine cannot take. Python and
// JavaScript both surface it as TypeError. Every message names the problem:
// not a number, not finite, not whole, or outside the int64 range.
struct IndexTypeError : std::invalid_argument {
    using std::invalid_argument::invalid_argument;
};

// An index that is not a number at all; `kind` names its type ("str", "string").
[[noreturn]] inline void throw_index_kind(const std::string& kind) {
    throw IndexTypeError("index must be a number (an integer, or a float with a "
                         "whole value), got " + kind);
}

[[noreturn]] inline void throw_index_out_of_range() {
    throw IndexTypeError("index is outside the int64 range [-2**63, 2**63); the "
                         "streaming engine is int64-indexed");
}

// A missing time (NumPy's NaT), which is NaN-like.
[[noreturn]] inline void throw_index_nat() {
    throw IndexTypeError("index must be finite, got NaT; the streaming engine "
                         "is int64-indexed");
}

// The error for a double that checked_index() refused.
[[noreturn]] inline void throw_bad_index(double d) {
    if (std::isnan(d))
        throw IndexTypeError("index must be finite, got NaN; the streaming engine "
                             "is int64-indexed");
    if (std::isinf(d))
        throw IndexTypeError("index must be finite, got infinity; the streaming "
                             "engine is int64-indexed");
    if (d != std::floor(d))
        throw IndexTypeError("index must be integer-valued; a fractional index was "
                             "seen. The streaming engine is int64-indexed (float "
                             "indices are not yet supported). Pass integer indices.");
    throw_index_out_of_range();
}

// The one rule for an index given as a double: finite, whole, inside int64.
// 2^63 is exactly representable; every whole double below it fits int64. NaN
// fails the range test.
inline std::int64_t checked_index(double d) {
    if (d >= -9223372036854775808.0 && d < 9223372036854775808.0 && d == std::floor(d))
        return static_cast<std::int64_t>(d);
    throw_bad_index(d);
}

// An unsigned integer index (a uint64 array element).
inline std::int64_t checked_index_u64(std::uint64_t u) {
    if (u > static_cast<std::uint64_t>(INT64_MAX)) throw_index_out_of_range();
    return static_cast<std::int64_t>(u);
}

}} // namespace screamer::dag
#endif
