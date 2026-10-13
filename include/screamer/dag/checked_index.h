#ifndef SCREAMER_DAG_CHECKED_INDEX_H
#define SCREAMER_DAG_CHECKED_INDEX_H

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <type_traits>

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

// The same for a stream merge whose index is a float: any int or float is
// taken there, fractional too.
[[noreturn]] inline void throw_float_index_kind(const std::string& kind) {
    throw IndexTypeError("index must be a number (an int or a float), got " + kind);
}

// A float-merge index given as an int too large for a double (beyond about
// 1.8e308; a smaller int rounds to the nearest double, as a batch float index
// array does).
[[noreturn]] inline void throw_float_index_overflow(std::size_t digits) {
    throw IndexTypeError("index does not fit a float64, got an int of " +
                         std::to_string(digits) + " digits");
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

// ---- Feed length -----------------------------------------------------------
// A (values, index) feed holds one index value per row of values. Every entry
// point that takes the pair checks this before it reads either array: a
// value error (ValueError in Python, RangeError in JavaScript) naming both
// lengths. `who` names the entry point ("Pipeline", "merge", "split"); `input`
// is the position of the feed, or -1 for an entry point with one stream.
[[noreturn]] inline void throw_feed_length(const std::string& who, std::ptrdiff_t input,
                                           std::size_t rows, std::size_t len) {
    const std::string tail = std::to_string(rows) + " rows of values but " +
                             std::to_string(len) + " index values";
    throw std::invalid_argument(who + ": " +
        (input >= 0 ? "input " + std::to_string(input) + " has " + tail : "the stream has " + tail));
}

inline void check_feed_length(const std::string& who, std::ptrdiff_t input,
                              std::size_t rows, std::size_t len) {
    if (rows != len) throw_feed_length(who, input, rows, len);
}

// ---- Index order -----------------------------------------------------------
// The index of every stream source must not decrease; equal indexes are
// allowed. A decrease is a value error (ValueError in Python, RangeError in
// JavaScript) that names the source and the event. One rule and one message
// for every stream entry point, checked where each source is read: the array
// and Python iterator readers (VectorSource, PySource) under every merge, the
// single-input batch loop and the live push of CompiledGraph, and split.

// The error paths stay out of line, so the per-event loops that check the
// order keep their size (functor_base.h defines the same macro).
#ifndef SCREAMER_NOINLINE
#if defined(__GNUC__) || defined(__clang__)
#define SCREAMER_NOINLINE __attribute__((noinline))
#elif defined(_MSC_VER)
#define SCREAMER_NOINLINE __declspec(noinline)
#else
#define SCREAMER_NOINLINE
#endif
#endif

namespace detail {
inline std::string index_text(std::int64_t k) { return std::to_string(k); }
inline std::string index_text(double k) {
    char buf[32];
    if (k == std::floor(k) && std::fabs(k) < 1e15) {   // a whole number, as written
        std::snprintf(buf, sizeof buf, "%.0f", k);
        return buf;
    }
    for (int prec = 1; prec <= 17; ++prec) {
        std::snprintf(buf, sizeof buf, "%.*g", prec, k);
        if (std::strtod(buf, nullptr) == k) break;
    }
    return buf;
}
}  // namespace detail

// `source` names the stream ("input 1", "the split stream"), `event` is the
// 0-based position of the event within that source.
template <class Index>
[[noreturn]] SCREAMER_NOINLINE void throw_index_decreases(const std::string& source,
                                                         std::size_t event, Index prev, Index now) {
    throw std::invalid_argument(
        "the index of " + source + " decreases at event " + std::to_string(event) +
        ", from " + detail::index_text(prev) + " to " + detail::index_text(now) +
        "; a stream's index must be non-decreasing (equal indexes are allowed)");
}

template <class Index>
[[noreturn]] SCREAMER_NOINLINE void throw_index_decreases(std::size_t input, std::size_t event,
                                                         Index prev, Index now) {
    throw_index_decreases("input " + std::to_string(input), event, prev, now);
}

// A float index of a stream merge or join must not be NaN: NaN has no place
// in the index order (every comparison with it is false), so it would pass
// the order check and give a row at index NaN. ValueError / RangeError.
[[noreturn]] SCREAMER_NOINLINE inline void throw_index_nan(std::size_t input, std::size_t event) {
    throw std::invalid_argument("the index of input " + std::to_string(input) + " is NaN at event " +
                                std::to_string(event) + "; an index must be a number");
}

// The value check of one index of input `input`: free for an integer index.
template <class Index>
inline void check_index_value(std::size_t input, std::size_t event, Index k) {
    if constexpr (std::is_floating_point_v<Index>) {
        if (k != k) throw_index_nan(input, event);
    }
}

}} // namespace screamer::dag
#endif
