#ifndef SCREAMER_DAG_RESAMPLE_PARAMS_H
#define SCREAMER_DAG_RESAMPLE_PARAMS_H

#include <cstddef>
#include <cstdint>
#include "screamer/common/eval_op.h"

namespace screamer { namespace dag {

enum class ResampleMode  { ByIndex, ByCount };
enum class ResampleAgg   { First, Last, Min, Max, Sum, Count, Mean, Ohlc };
enum class ResampleLabel { Left, Right };
// Empty-window fill policy for internal gaps (buckets with no events between two
// events). Skip = no row (default, legacy behavior); Nan = an all-NaN row at the
// gap's label; Carry = repeat the previous emitted row's values verbatim.
enum class ResampleFill  { Skip, Nan, Carry };

struct ResampleParams {
    ResampleMode  mode  = ResampleMode::ByIndex;
    ResampleAgg   agg   = ResampleAgg::Last;
    ResampleLabel label = ResampleLabel::Left;
    ResampleFill  fill  = ResampleFill::Skip;
    std::int64_t  width  = 1;   // ByIndex
    std::int64_t  origin = 0;   // ByIndex
    std::int64_t  count  = 1;   // ByCount
    // When non-null, the bucket reducer is this arbitrary functor (GenericResample
    // path) instead of the builtin `agg` enum. The pointee is owned by Python; the
    // graph builder holds a py::object ref so it outlives the compiled graph.
    EvalOp* reducer = nullptr;
};

inline std::size_t resample_width(ResampleAgg a) {
    return a == ResampleAgg::Ohlc ? 4u : 1u;
}

// Emitted-frame width for a resample node: the reducer's arity when a functor
// reducer is present, otherwise the builtin agg width.
inline std::size_t resample_output_width(const ResampleParams& p) {
    return p.reducer ? p.reducer->n_out() : resample_width(p.agg);
}

}} // namespace screamer::dag
#endif
