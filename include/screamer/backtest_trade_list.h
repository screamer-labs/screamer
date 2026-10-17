#ifndef SCREAMER_BACKTEST_TRADE_LIST_H
#define SCREAMER_BACKTEST_TRADE_LIST_H

// backtest_trades: the round trips of a backtest engine's output, one per
// trade, as BacktestStats counts them (detail::RoundTripTracker). Shared by
// the Python backtest_trades and the JavaScript backtestTrades, which only
// label the result. Rows with a NaN field are skipped. A trade still open on
// the last row is listed with exit = T - 1 and closed = false.

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>
#include "screamer/backtest_summary.h"
#include "screamer/detail/round_trips.h"

namespace screamer {

    // One asset's (T, 4) output; a portfolio's (T, K, 4) has one trade list
    // per asset.
    inline void check_trade_rows(std::size_t ndim, std::size_t ncols) {
        if (ndim == 3)
            throw std::invalid_argument(
                "backtest_trades takes one asset's (T, 4) output; call it once per asset "
                "of a (T, K, 4) portfolio output");
        check_engine_rows(ndim, ncols);
    }

    inline std::vector<detail::RoundTrip> backtest_trades(const double* eng, std::size_t T,
                                                          std::ptrdiff_t row_stride,
                                                          std::ptrdiff_t col_stride) {
        std::vector<detail::RoundTrip> out;
        detail::RoundTripTracker tracker;
        for (std::size_t t = 0; t < T; ++t) {
            const double* r = eng + static_cast<std::ptrdiff_t>(t) * row_stride;
            const double equity = r[0], pnl = r[col_stride], position = r[2 * col_stride],
                         cost = r[3 * col_stride];
            if (std::isnan(equity) || std::isnan(pnl) || std::isnan(position) || std::isnan(cost))
                continue;
            tracker.step(t, pnl, cost, position,
                         [&out](const detail::RoundTrip& tr) { out.push_back(tr); });
        }
        if (tracker.open()) {
            detail::RoundTrip tr = tracker.current();
            tr.exit = T - 1;
            tr.closed = false;
            out.push_back(tr);
        }
        return out;
    }

} // namespace screamer

#endif // SCREAMER_BACKTEST_TRADE_LIST_H
