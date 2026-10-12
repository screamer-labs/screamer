#ifndef SCREAMER_DAG_CHECKED_INDEX_H
#define SCREAMER_DAG_CHECKED_INDEX_H

#include <cmath>
#include <cstdint>
#include <stdexcept>

namespace screamer { namespace dag {

// The kind error for an index the int64-indexed engine cannot take. Python and
// JavaScript both surface it as TypeError.
struct IndexTypeError : std::invalid_argument {
    IndexTypeError()
        : std::invalid_argument(
              "index must be integer-valued; a fractional index was seen. The "
              "streaming engine is int64-indexed (float indices are not yet "
              "supported). Pass integer indices.") {}
};

// The one rule for an index given as a double: finite, whole, inside int64.
inline std::int64_t checked_index(double d) {
    // 2^63 is exactly representable; every double below it fits int64.
    if (!(std::isfinite(d) && d == std::floor(d) &&
          d >= -9223372036854775808.0 && d < 9223372036854775808.0))
        throw IndexTypeError();
    return static_cast<std::int64_t>(d);
}

}} // namespace screamer::dag
#endif
