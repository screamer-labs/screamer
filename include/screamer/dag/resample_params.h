#ifndef SCREAMER_DAG_RESAMPLE_PARAMS_H
#define SCREAMER_DAG_RESAMPLE_PARAMS_H

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>
#include "screamer/common/checked_arith.h"
#include "screamer/common/eval_op.h"

namespace screamer { namespace dag {

enum class ResampleMode  { ByIndex, ByCount, ByCumulative, ByClock };
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

// The most empty bars a fill Resample (fill nan or carry, every=) emits for
// one gap: the empty buckets that one event, or one advance(now), closes at
// once. A larger gap raises before any bar is built, so a gap of 2**53
// buckets (an index in nanoseconds with every=10, or a long pause before an
// advance) fails at once instead of exhausting memory. Each fill bar costs at
// least 16 bytes in the graph (its index and one value) and about as much
// again in the result, so the default bounds one gap near 32 MB for one
// column: 1e6 one-second bars cover 11 days, 1e6 one-minute bars 1.9 years.
// Settable per Resample (max_fill, maxFill in JavaScript) from 1 to 2**53.
inline constexpr std::int64_t kDefaultMaxFill = 1'000'000;

// The fill-limit error: `bars` empty bars for one gap (-1: beyond int64).
[[noreturn]] inline void throw_fill_limit(std::int64_t bars, std::int64_t max_fill) {
    throw std::invalid_argument(
        "a fill Resample would emit " +
        (bars < 0 ? std::string("more than 2**63") : std::to_string(bars)) +
        " empty bars for one gap, more than max_fill=" + std::to_string(max_fill) +
        " (maxFill in JavaScript); raise max_fill, or check that every= and the "
        "index have the same unit");
}

// Checks the empty bars that closing bucket `from` and moving to bucket `to`
// emits: the buckets strictly between them, plus `from` itself when it is
// empty. Runs before any bar is pushed.
inline void check_fill_gap(std::int64_t from, std::int64_t to, bool from_empty,
                           std::int64_t max_fill) {
    std::int64_t d;
    if (!checked_sub(to, from, d)) throw_fill_limit(-1, max_fill);
    const std::int64_t bars = d - 1 + (from_empty ? 1 : 0);
    if (bars > max_fill) throw_fill_limit(bars, max_fill);
}

struct ResampleParams {
    ResampleMode  mode  = ResampleMode::ByIndex;
    ResampleAgg   agg   = ResampleAgg::Last;
    ResampleLabel label = ResampleLabel::Left;
    ResampleFill  fill  = ResampleFill::Skip;
    std::int64_t  width  = 1;   // ByIndex
    std::int64_t  origin = 0;   // ByIndex
    std::int64_t  count  = 1;   // ByCount
    double        threshold = 0.0; // ByCumulative: close when cum_driver >= threshold
    // ByClock as-of freshness limit in index units; -1 disables expiry.
    std::int64_t  max_age = -1;
    // ByIndex with a fill: the most empty bars one gap may emit.
    std::int64_t  max_fill = kDefaultMaxFill;
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

// resample_width(OhlcvBars)==0 is a sentinel meaning "width deferred to first push":
// the actual plan (and output width) is built from the runtime input width at node-init.
inline std::size_t resample_width(ResampleAgg a) {
    if (a == ResampleAgg::Ohlc) return 4u;
    if (a == ResampleAgg::OhlcvBars) return 0u;  // sentinel: width deferred to first push
    return 1u;
}

// Number of input columns required by the plan (max input_col + 1, or 0).
inline std::size_t resample_plan_input_cols(const std::vector<ResamplePlanEntry>& plan) {
    std::size_t max_col = 0;
    for (const auto& e : plan) if (e.input_col > max_col) max_col = e.input_col;
    return plan.empty() ? 0u : max_col + 1u;
}

// ohlcv_bars reads [open, high, low, close, volume, ...]: at least one volume
// column (docs/functions_streams/Resample.md). One rule for every path: the
// Pipeline node, the eager resample() and the lazy one.
constexpr std::size_t kOhlcvBarsMinColumns = 5;

inline void check_ohlcv_bars_width(std::size_t W) {
    if (W < kOhlcvBarsMinColumns)
        throw std::invalid_argument(
            "ohlcv_bars takes [open, high, low, close, volume, ...] (at least 5 "
            "columns), but its input has " + std::to_string(W));
}

// Build the ohlcv_bars plan for a given input width W (at least 5; throws
// std::invalid_argument otherwise).
// Plan: first(col0), max(col1), min(col2), last(col3), sum(col4), sum(col5), ...
inline std::vector<ResamplePlanEntry> make_ohlcv_bars_plan(std::size_t W) {
    check_ohlcv_bars_width(W);
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

// Per-column plan for a builtin single-column agg on a W-column stream: every
// column is aggregated on its own, and Ohlc gives four
// outputs per column, (open, high, low, close).
inline std::vector<ResamplePlanEntry> make_per_column_plan(ResampleAgg agg, std::size_t W) {
    std::vector<ResamplePlanEntry> plan;
    plan.reserve(agg == ResampleAgg::Ohlc ? 4 * W : W);
    for (std::size_t c = 0; c < W; ++c) {
        if (agg == ResampleAgg::Ohlc) {
            plan.push_back({ResampleAgg::First, c});
            plan.push_back({ResampleAgg::Max,   c});
            plan.push_back({ResampleAgg::Min,   c});
            plan.push_back({ResampleAgg::Last,  c});
        } else {
            plan.push_back({agg, c});
        }
    }
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
