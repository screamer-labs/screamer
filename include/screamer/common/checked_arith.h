#ifndef SCREAMER_COMMON_CHECKED_ARITH_H
#define SCREAMER_COMMON_CHECKED_ARITH_H

#include <cstdint>
#include <limits>

namespace screamer {

// Portable int64 arithmetic that never overflows. Each operation checks the
// range with plain comparisons before it adds or subtracts, so it compiles on
// every compiler (GCC, Clang, MSVC, Emscripten) without builtins.

// a + b. Returns false, and leaves `out` untouched, when the sum is outside the
// int64 range.
inline bool checked_add(std::int64_t a, std::int64_t b, std::int64_t& out) {
    if (b > 0 ? a > std::numeric_limits<std::int64_t>::max() - b
              : a < std::numeric_limits<std::int64_t>::min() - b)
        return false;
    out = a + b;
    return true;
}

// a - b. Returns false, and leaves `out` untouched, when the difference is
// outside the int64 range.
inline bool checked_sub(std::int64_t a, std::int64_t b, std::int64_t& out) {
    if (b < 0 ? a > std::numeric_limits<std::int64_t>::max() + b
              : a < std::numeric_limits<std::int64_t>::min() + b)
        return false;
    out = a - b;
    return true;
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
