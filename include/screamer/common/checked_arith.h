#ifndef SCREAMER_COMMON_CHECKED_ARITH_H
#define SCREAMER_COMMON_CHECKED_ARITH_H

#include <cstdint>
#include <limits>

namespace screamer {

// Portable int64 arithmetic that never overflows. GCC and Clang (Emscripten
// included) use their overflow builtins (one flag test, no division); other
// compilers (MSVC) check the range with plain comparisons first.

// a + b. Returns false, and leaves `out` untouched, when the sum is outside the
// int64 range.
inline bool checked_add(std::int64_t a, std::int64_t b, std::int64_t& out) {
#if defined(__GNUC__) || defined(__clang__)
    std::int64_t r;
    if (__builtin_add_overflow(a, b, &r)) return false;
    out = r;
    return true;
#else
    if (b > 0 ? a > std::numeric_limits<std::int64_t>::max() - b
              : a < std::numeric_limits<std::int64_t>::min() - b)
        return false;
    out = a + b;
    return true;
#endif
}

// a - b. Returns false, and leaves `out` untouched, when the difference is
// outside the int64 range.
inline bool checked_sub(std::int64_t a, std::int64_t b, std::int64_t& out) {
#if defined(__GNUC__) || defined(__clang__)
    std::int64_t r;
    if (__builtin_sub_overflow(a, b, &r)) return false;
    out = r;
    return true;
#else
    if (b < 0 ? a > std::numeric_limits<std::int64_t>::max() + b
              : a < std::numeric_limits<std::int64_t>::min() + b)
        return false;
    out = a - b;
    return true;
#endif
}

// a * b. Returns false, and leaves `out` untouched, when the product is
// outside the int64 range.
inline bool checked_mul(std::int64_t a, std::int64_t b, std::int64_t& out) {
#if defined(__GNUC__) || defined(__clang__)
    std::int64_t r;
    if (__builtin_mul_overflow(a, b, &r)) return false;
    out = r;
    return true;
#else
    constexpr std::int64_t hi = std::numeric_limits<std::int64_t>::max();
    constexpr std::int64_t lo = std::numeric_limits<std::int64_t>::min();
    if (a > 0) {
        if (b > 0 ? a > hi / b : b < lo / a) return false;
    } else if (a < 0) {
        if (b > 0 ? a < lo / b : (b != 0 && a < hi / b)) return false;
    }
    out = a * b;
    return true;
#endif
}

// a + b, clamped to the int64 range.
inline std::int64_t saturating_add(std::int64_t a, std::int64_t b) {
    std::int64_t out;
    if (checked_add(a, b, out)) return out;
    return b > 0 ? std::numeric_limits<std::int64_t>::max()
                 : std::numeric_limits<std::int64_t>::min();
}

// a - b, clamped to the int64 range.
inline std::int64_t saturating_sub(std::int64_t a, std::int64_t b) {
    std::int64_t out;
    if (checked_sub(a, b, out)) return out;
    return b < 0 ? std::numeric_limits<std::int64_t>::max()
                 : std::numeric_limits<std::int64_t>::min();
}

} // namespace screamer
#endif
