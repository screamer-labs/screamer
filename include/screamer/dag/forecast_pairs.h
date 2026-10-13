#ifndef SCREAMER_DAG_FORECAST_PAIRS_H
#define SCREAMER_DAG_FORECAST_PAIRS_H

// The forecast_pairs graph: features delayed by a horizon, paired with a
// target. One definition serves Python's forecast_pairs and JavaScript's
// forecastPairs in every regime (eager, lazy, live, and a graph node), with
// the rules both bindings check.
//
// Count mode lags the rows of X by `count` events (LagRowsNode, any width).
// Duration mode delays X's index by `duration` index units and samples the
// last delayed X at each y event: an as-of join driven by the y clock,
// Resample(Delay(duration)(X), clock=y, agg="last", fill="carry"). Both then
// pair the shifted X with y through CombineLatest. The output row is
// [X_shifted..., y]; dropna drops a row that holds a NaN.

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

// The text of a y that is not one column.
inline std::string forecast_pairs_y_text() { return "forecast_pairs: y"; }

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
        throw std::invalid_argument("forecast_pairs: X has no feature columns");
    if (y_ndim > 2)
        throw std::invalid_argument("forecast_pairs: y has " + std::to_string(y_ndim) +
                                    " dimensions, expected (T,) or (T, 1)");
    if (y_ndim == 2 && y_cols != 1)
        throw std::invalid_argument(forecast_pairs_y_text() + " holds " + std::to_string(y_cols) +
                                    " columns, expected 1");
    if (count_mode && x_rows != y_rows)
        throw std::invalid_argument("X and y must share the same length (time axis)");
}

// Adds the forecast-pairs nodes that read X from node `x` (any width) and y
// from node `y` (one column, checked when the widths are known: the join
// requires it); returns the output node. Exactly one of `count` and `duration` is given.
inline std::size_t add_forecast_pairs(GraphBuilder& gb, std::size_t x, std::size_t y,
                                      std::optional<std::int64_t> count,
                                      std::optional<std::int64_t> duration, bool dropna) {
    check_forecast_pairs_horizon(count, duration);
    std::size_t shifted;
    if (count) {
        shifted = gb.add_lag_rows({x}, static_cast<std::size_t>(*count));
    } else {
        ResampleParams rp;
        rp.mode = ResampleMode::ByClock;
        rp.agg  = ResampleAgg::Last;
        rp.fill = ResampleFill::Carry;
        shifted = gb.add_resample({gb.add_delay({x}, *duration), y}, rp);
    }
    const std::size_t pairs = gb.add_combine_latest({shifted, y}, true);
    gb.require_width(pairs, 1, 1, forecast_pairs_y_text());
    return dropna ? gb.add_dropna({pairs}, false) : pairs;
}

// The whole graph: inputs X and y (in that order), one output.
inline GraphSpec forecast_pairs_graph(std::optional<std::int64_t> count,
                                      std::optional<std::int64_t> duration, bool dropna) {
    GraphBuilder gb;
    const std::size_t x = gb.add_input();
    const std::size_t y = gb.add_input();
    gb.set_outputs({add_forecast_pairs(gb, x, y, count, duration, dropna)});
    return gb.take();
}

}}  // namespace screamer::dag
#endif
