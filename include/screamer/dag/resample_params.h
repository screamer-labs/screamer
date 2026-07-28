#ifndef SCREAMER_DAG_RESAMPLE_PARAMS_H
#define SCREAMER_DAG_RESAMPLE_PARAMS_H

#include <cstddef>
#include <cstdint>
#include <vector>
#include "screamer/common/eval_op.h"

namespace screamer { namespace dag {

enum class ResampleMode  { ByIndex, ByCount };
// ResampleAgg enumerates single-column reducer kinds.
// Each output column in a plan has one ResampleAgg applied to one input column.
// SumPos: sum of max(v, 0) per non-NaN value (= total positive part).
// SumNeg: sum of |min(v, 0)| per non-NaN value (= total magnitude of negative
//   part, always non-negative). Matches NegPart() + sum used in ohlcv2.
// OhlcvBars: a dynamic multi-column bar agg - plan is built at node-instantiation
//   time from the actual input width (first 4 cols OHLC via first/max/min/last,
//   trailing cols summed). Width must be >= 5. Not a plan entry type; a special
//   sentinel that triggers dynamic plan creation in the compiled graph.
enum class ResampleAgg   { First, Last, Min, Max, Sum, Count, Mean, Ohlc,
                           SumPos, SumNeg, OhlcvBars };
enum class ResampleLabel { Left, Right };
// Empty-window fill policy for internal gaps (buckets with no events between two
// events). Skip = no row (default, legacy behavior); Nan = an all-NaN row at the
// gap's label; Carry = repeat the previous emitted row's values verbatim.
enum class ResampleFill  { Skip, Nan, Carry };

// One entry in a per-column reducer plan. Maps an output column to a reducer
// applied to a specific input column.
struct ResamplePlanEntry {
    ResampleAgg  agg;        // reducer kind for this output column
    std::size_t  input_col;  // which input column to reduce (0-based)
};

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
    // Per-column reducer plan for multi-column bar aggs (ohlc_bars, ohlcv_bars,
    // ohlcv, ohlcv2). When plan is non-empty, each entry specifies the reducer
    // kind and input column index for one output column, and agg is ignored.
    // plan.size() is the output width. num_input_cols() is the count of distinct
    // input columns required.
    std::vector<ResamplePlanEntry> plan;
};

inline std::size_t resample_width(ResampleAgg a) {
    // OhlcvBars width is dynamic (equals input width); return 0 as sentinel.
    if (a == ResampleAgg::Ohlc) return 4u;
    if (a == ResampleAgg::OhlcvBars) return 0u;  // dynamic; must use plan
    return 1u;
}

// Number of input columns required by the plan (max input_col + 1, or 0).
inline std::size_t resample_plan_input_cols(const std::vector<ResamplePlanEntry>& plan) {
    std::size_t max_col = 0;
    for (const auto& e : plan) if (e.input_col > max_col) max_col = e.input_col;
    return plan.empty() ? 0u : max_col + 1u;
}

// Build the ohlcv_bars plan for a given input width W (must be >= 5).
// Plan: first(col0), max(col1), min(col2), last(col3), sum(col4), sum(col5), ...
inline std::vector<ResamplePlanEntry> make_ohlcv_bars_plan(std::size_t W) {
    std::vector<ResamplePlanEntry> plan;
    plan.reserve(W);
    plan.push_back({ResampleAgg::First, 0});
    plan.push_back({ResampleAgg::Max,   1});
    plan.push_back({ResampleAgg::Min,   2});
    plan.push_back({ResampleAgg::Last,  3});
    for (std::size_t c = 4; c < W; ++c)
        plan.push_back({ResampleAgg::Sum, c});
    return plan;
}

// Emitted-frame width for a resample node: the plan width when a plan is given;
// the reducer's arity when a functor reducer is present; otherwise the builtin
// single-column agg width.
inline std::size_t resample_output_width(const ResampleParams& p) {
    if (!p.plan.empty()) return p.plan.size();
    return p.reducer ? p.reducer->n_out() : resample_width(p.agg);
}

}} // namespace screamer::dag
#endif
