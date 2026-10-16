#ifndef SCREAMER_DAG_FORECAST_PAIRS_H
#define SCREAMER_DAG_FORECAST_PAIRS_H

// The forecast_pairs graph: features paired with a target over a horizon
// after them. One definition serves Python's forecast_pairs and JavaScript's
// forecastPairs in every regime (eager, lazy, live, and a graph node), with
// the rules both bindings check.
//
// `index` chooses whose index a row carries.
//
// index="features": the row for the features at t holds the target over
// (t, t + horizon], stamped t, and it is emitted once that target is known.
// Count mode joins X and y into rows (CombineLatest) and pairs each row with
// the target `count` rows later (LeadRowsNode). Duration mode delays y's index
// by -duration and joins it as-of onto X's clock (ForwardAsOfNode): the last y
// at or before t + duration, NaN when y ends before t + duration. The rows
// still waiting at the end of input get a NaN target.
//
// index="target": the row is stamped at the target. Count mode lags the rows
// of X by `count` events (LagRowsNode). Duration mode delays X's index by
// `duration` index units and samples the last delayed X at each y event: an
// as-of join driven by the y clock, Resample(Delay(duration)(X), clock=y,
// agg="last", fill="carry"). Both then pair the shifted X with y through
// CombineLatest.
//
// The output row is [X..., y]; dropna drops a row that holds a NaN.

#include <climits>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include "screamer/dag/graph.h"
#include "screamer/dag/resample_params.h"

namespace screamer { namespace dag {

// A count (from 1 to INT_MAX) or duration (from 0 to the int64 maximum) out of
// range; `text` is the value as the binding writes it.
[[noreturn]] inline void throw_forecast_pairs_range(const std::string& name,
                                                    const std::string& text) {
    if (name == "count")
        throw std::invalid_argument("forecast_pairs: count must be at least 1 and at most " +
                                    std::to_string(INT_MAX) + ", got " + text);
    throw std::invalid_argument("forecast_pairs: " + name + " must be at least 0 and at most " +
                                std::to_string(INT64_MAX) + ", got " + text);
}

// The horizon rule: exactly one of `count` and `duration`, each in range. A
// negative duration would pair y with later X, so it is refused.
inline void check_forecast_pairs_horizon(std::optional<std::int64_t> count,
                                         std::optional<std::int64_t> duration) {
    if (count.has_value() == duration.has_value())
        throw std::invalid_argument("pass exactly one of count= or duration=");
    if (count && (*count < 1 || *count > INT_MAX))
        throw_forecast_pairs_range("count", std::to_string(*count));
    if (duration && *duration < 0)
        throw_forecast_pairs_range("duration", std::to_string(*duration));
}

// The index setting: true for "features", false for "target".
inline bool forecast_pairs_features_index(const std::string& index) {
    if (index == "features") return true;
    if (index == "target") return false;
    throw std::invalid_argument("forecast_pairs: index must be \"features\" or \"target\", got '" +
                                index + "'");
}

// The rules on X (at least one feature column) and y (one column).
inline WidthRule forecast_pairs_x_rule(std::size_t slot) {
    WidthRule r; r.slot = slot; r.min = 1; r.what = "forecast_pairs: X"; return r;
}
inline WidthRule forecast_pairs_y_rule(std::size_t slot) {
    WidthRule r; r.slot = slot; r.exact = 1; r.min = 1; r.what = "forecast_pairs: y"; return r;
}

// The input rule of an eager call: X is (T,) or (T, K) with K >= 1, y is one
// column ((T,) or (T, 1)), and in count mode, which pairs row t with row t,
// both have T rows.
inline void check_forecast_pairs_inputs(bool count_mode, std::size_t x_ndim, std::size_t x_rows,
                                        std::size_t x_cols, std::size_t y_ndim,
                                        std::size_t y_rows, std::size_t y_cols) {
    if (x_ndim > 2)
        throw std::invalid_argument("forecast_pairs: X has " + std::to_string(x_ndim) +
                                    " dimensions, expected (T,) or (T, K)");
    if (x_ndim == 2 && x_cols == 0)
        throw std::invalid_argument(width_rule_text(forecast_pairs_x_rule(0), 0));
    if (y_ndim > 2)
        throw std::invalid_argument("forecast_pairs: y has " + std::to_string(y_ndim) +
                                    " dimensions, expected (T,) or (T, 1)");
    if (y_ndim == 2 && y_cols != 1)
        throw std::invalid_argument(width_rule_text(forecast_pairs_y_rule(0), y_cols));
    if (count_mode && x_rows != y_rows)
        throw std::invalid_argument("X and y must share the same length (time axis)");
}

// Adds the forecast-pairs nodes that read X from node `x` (any width) and y
// from node `y` (one column); returns the output node. Width rules on the
// nodes that read X and y check both before any event is fed. Exactly one of
// `count` and `duration` is given. `features_index` stamps the rows at the
// features (index="features"), else at the target (index="target").
inline std::size_t add_forecast_pairs(GraphBuilder& gb, std::size_t x, std::size_t y,
                                      std::optional<std::int64_t> count,
                                      std::optional<std::int64_t> duration, bool dropna,
                                      bool features_index) {
    check_forecast_pairs_horizon(count, duration);
    std::size_t pairs;
    if (features_index) {
        if (count) {
            const std::size_t rows = gb.add_combine_latest({x, y}, true);
            gb.require_width(rows, forecast_pairs_x_rule(0));
            gb.require_width(rows, forecast_pairs_y_rule(1));
            pairs = gb.add_lead_rows({rows}, static_cast<std::size_t>(*count));
        } else {
            const std::size_t ahead = gb.add_delay({y}, -*duration);
            pairs = gb.add_forward_asof({x, ahead});
            gb.require_width(pairs, forecast_pairs_x_rule(0));
            gb.require_width(pairs, forecast_pairs_y_rule(1));
        }
    } else {
        std::size_t shifted;
        if (count) {
            shifted = gb.add_lag_rows({x}, static_cast<std::size_t>(*count));
            gb.require_width(shifted, forecast_pairs_x_rule(0));
        } else {
            ResampleParams rp;
            rp.mode = ResampleMode::ByClock;
            rp.agg  = ResampleAgg::Last;
            rp.fill = ResampleFill::Carry;
            const std::size_t delayed = gb.add_delay({x}, *duration);
            gb.require_width(delayed, forecast_pairs_x_rule(0));
            shifted = gb.add_resample({delayed, y}, rp);
        }
        pairs = gb.add_combine_latest({shifted, y}, true);
        gb.require_width(pairs, forecast_pairs_y_rule(1));
    }
    return dropna ? gb.add_dropna({pairs}, false) : pairs;
}

// The whole graph: inputs X and y (in that order), one output.
inline GraphSpec forecast_pairs_graph(std::optional<std::int64_t> count,
                                      std::optional<std::int64_t> duration, bool dropna,
                                      bool features_index) {
    GraphBuilder gb;
    const std::size_t x = gb.add_input();
    const std::size_t y = gb.add_input();
    gb.set_outputs({add_forecast_pairs(gb, x, y, count, duration, dropna, features_index)});
    return gb.take();
}

}}  // namespace screamer::dag
#endif
