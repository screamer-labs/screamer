#ifndef SCREAMER_DAG_RESAMPLE_PARAMS_H
#define SCREAMER_DAG_RESAMPLE_PARAMS_H

#include <cstddef>
#include <cstdint>

namespace screamer { namespace dag {

enum class ResampleMode  { ByKey, ByCount };
enum class ResampleAgg   { First, Last, Min, Max, Sum, Count, Mean, Ohlc };
enum class ResampleLabel { Left, Right };

struct ResampleParams {
    ResampleMode  mode  = ResampleMode::ByKey;
    ResampleAgg   agg   = ResampleAgg::Last;
    ResampleLabel label = ResampleLabel::Left;
    std::int64_t  width  = 1;   // ByKey
    std::int64_t  origin = 0;   // ByKey
    std::int64_t  count  = 1;   // ByCount
};

inline std::size_t resample_width(ResampleAgg a) {
    return a == ResampleAgg::Ohlc ? 4u : 1u;
}

}} // namespace screamer::dag
#endif
