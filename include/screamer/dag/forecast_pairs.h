#ifndef SCREAMER_DAG_FORECAST_PAIRS_H
#define SCREAMER_DAG_FORECAST_PAIRS_H

// The forecast_pairs graph: features delayed by a horizon, paired with a
// target. One definition serves Python's forecast_pairs and JavaScript's
// forecastPairs in every regime (eager, lazy, live, and a graph node).
//
// Count mode lags X by `count` events: Lag(count) on each feature column,
// then CombineLatest with y. Duration mode delays X's index by `duration`
// index units and samples the last delayed X at each y event: an as-of join
// driven by the y clock, Resample(Delay(duration)(X), clock=y, agg="last",
// fill="carry"), then CombineLatest with y. The output row is
// [X_shifted..., y]; dropna drops a row that holds a NaN.

#include <climits>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include "screamer/dag/graph.h"
#include "screamer/dag/resample_params.h"
#include "screamer/lag.h"

namespace screamer { namespace dag {

// An eager call pairs row t of X with row t of y, so both have T rows.
inline void check_forecast_pairs_rows(std::size_t x_rows, std::size_t y_rows) {
    if (x_rows != y_rows)
        throw std::invalid_argument("X and y must share the same length (time axis)");
}

// The horizon rule: exactly one of `count` (events, from 1 to INT_MAX) and
// `duration` (index units, any int64) is given.
inline void check_forecast_pairs_horizon(std::optional<std::int64_t> count,
                                         std::optional<std::int64_t> duration) {
    if (count.has_value() == duration.has_value())
        throw std::invalid_argument("pass exactly one of count= or duration=");
    if (count && (*count < 1 || *count > INT_MAX))
        throw std::invalid_argument("forecast_pairs: count must be at least 1 and at most " +
                                    std::to_string(INT_MAX) + ", got " +
                                    std::to_string(*count));
}

// Adds the forecast-pairs nodes that read X from node `x` (`n_features`
// columns) and y from node `y`; returns the output node. Exactly one of
// `count` (events, at least 1) and `duration` (index units) is given.
// `n_features` shapes count mode only: Lag takes one column, so a wide X is
// split into its columns first. Duration mode takes X of any width.
inline std::size_t add_forecast_pairs(GraphBuilder& gb, std::size_t x, std::size_t y,
                                      std::size_t n_features,
                                      std::optional<std::int64_t> count,
                                      std::optional<std::int64_t> duration, bool dropna) {
    check_forecast_pairs_horizon(count, duration);
    std::vector<std::size_t> row;
    if (count) {
        if (n_features < 1)
            throw std::invalid_argument("forecast_pairs: X has no feature columns");
        const int lag = static_cast<int>(*count);
        if (n_features == 1) {
            row.push_back(gb.add_owned_functor(std::make_shared<Lag>(lag), {x}));
        } else {
            row.reserve(n_features + 1);
            for (std::size_t c = 0; c < n_features; ++c)
                row.push_back(gb.add_owned_functor(std::make_shared<Lag>(lag),
                                                   {gb.add_select({x}, {c})}));
        }
    } else {
        ResampleParams rp;
        rp.mode = ResampleMode::ByClock;
        rp.agg  = ResampleAgg::Last;
        rp.fill = ResampleFill::Carry;
        row.push_back(gb.add_resample({gb.add_delay({x}, *duration), y}, rp));
    }
    row.push_back(y);
    const std::size_t pairs = gb.add_combine_latest(std::move(row), true);
    return dropna ? gb.add_dropna({pairs}, false) : pairs;
}

// The whole graph: inputs X and y (in that order), one output.
inline GraphSpec forecast_pairs_graph(std::size_t n_features,
                                      std::optional<std::int64_t> count,
                                      std::optional<std::int64_t> duration, bool dropna) {
    GraphBuilder gb;
    const std::size_t x = gb.add_input();
    const std::size_t y = gb.add_input();
    gb.set_outputs({add_forecast_pairs(gb, x, y, n_features, count, duration, dropna)});
    return gb.take();
}

}}  // namespace screamer::dag
#endif
