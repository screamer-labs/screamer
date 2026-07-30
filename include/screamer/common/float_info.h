#ifndef SCREAMER_FLOAT_INFO_H
#define SCREAMER_FLOAT_INFO_H

#include <cstdint>   // uint64_t / uint32_t used below (self-contained header)
#include <cstring>   // std::memcpy

/*
we use -ffast_math, but...

-ffast-math
Sets the options -fno-math-errno, -funsafe-math-optimizations, -ffinite-math-only, -fno-rounding-math, 
-fno-signaling-nans, -fcx-limited-range and -fexcess-precision=fast.

This option causes the preprocessor macro __FAST_MATH__ to be defined.

This option is not turned on by any -O option besides -Ofast since it can result in incorrect output 
for programs that depend on an exact implementation of IEEE or ISO rules/specifications for math 
functions. It may, however, yield faster code for programs that do not require the guarantees of these 
specifications.

The code below provided NaN tests that DO work with -ffast_math
Code below is copied from Maxim Egorushkin on https://stackoverflow.com/a/57770634
Thanks Maxim!
*/

namespace screamer {

    static inline uint64_t load_ieee754_rep(double a) {
        uint64_t r;
        static_assert(sizeof r == sizeof a, "Unexpected sizes.");
        std::memcpy(&r, &a, sizeof a); // Generates movq instruction.
        return r;
    }

    static inline uint32_t load_ieee754_rep(float a) {
        uint32_t r;
        static_assert(sizeof r == sizeof a, "Unexpected sizes.");
        std::memcpy(&r, &a, sizeof a); // Generates movd instruction.
        return r;
    }

    constexpr uint64_t inf_double_shl1 = UINT64_C(0xffe0000000000000);
    constexpr uint32_t inf_float_shl1 = UINT32_C(0xff000000);

    // The shift left removes the sign bit. The exponent moves into the topmost bits,
    // so that plain unsigned comparison is enough.
    static inline bool isnan2(double a)    { return load_ieee754_rep(a) << 1  > inf_double_shl1; }
    static inline bool isinf2(double a)    { return load_ieee754_rep(a) << 1 == inf_double_shl1; }
    static inline bool isfinite2(double a) { return load_ieee754_rep(a) << 1  < inf_double_shl1; }
    static inline bool isnan2(float a)     { return load_ieee754_rep(a) << 1  > inf_float_shl1; }
    static inline bool isinf2(float a)     { return load_ieee754_rep(a) << 1 == inf_float_shl1; }
    static inline bool isfinite2(float a)  { return load_ieee754_rep(a) << 1  < inf_float_shl1; }

    // Multi-input NaN test for operators under the `ignore` policy, where one
    // NaN field makes the whole bar unusable. Bitwise `|` rather than `||`:
    // isnan2 is integer-only (a shift and a compare, no FP unit), so folding
    // the results costs less than the extra branches short-circuiting would
    // add. One branch at the call site instead of two or three.
    static inline bool any_nan(double a, double b) {
        return isnan2(a) | isnan2(b);
    }
    static inline bool any_nan(double a, double b, double c) {
        return isnan2(a) | isnan2(b) | isnan2(c);
    }
    static inline bool any_nan(double a, double b, double c, double d) {
        return isnan2(a) | isnan2(b) | isnan2(c) | isnan2(d);
    }

} // namespace

#endif