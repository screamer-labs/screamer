#ifndef SCREAMER_DETAIL_ROUND_TRIPS_H
#define SCREAMER_DETAIL_ROUND_TRIPS_H

// The round trips of a backtest engine's position column, shared by
// BacktestStats (its trade statistics) and backtest_trades (the trade list),
// so the two count the same trades. A trade opens when the position leaves
// zero and closes when it returns to zero; a sign flip closes one trade and
// opens the next on the same row. Resizing without crossing zero stays one
// trade. A trade's pnl and cost are the sums from its opening row to its
// closing row; on a flip row they go to the closing trade, because the
// engine reports one pnl per row. `held` counts the rows that held a
// position: every row from the opening row up to the row before the close.

#include <cstddef>

namespace screamer { namespace detail {

struct RoundTrip {
    std::size_t entry = 0, exit = 0;
    double side = 0.0;              // +1 long, -1 short (the opening position's sign)
    double pnl = 0.0, cost = 0.0;
    long long held = 0;
    bool closed = false;
};

class RoundTripTracker {
public:
    void reset() {
        open_ = false;
        prev_pos_ = 0.0;
        cur_ = RoundTrip{};
    }

    // One row without NaN; on_close(trade) for a trade that closes on it.
    template <class OnClose>
    void step(std::size_t row, double pnl, double cost, double position, OnClose&& on_close) {
        const bool was_open = open_;
        const bool flipped = prev_pos_ != 0.0 && position != 0.0 &&
                             ((prev_pos_ > 0.0) != (position > 0.0));
        if (was_open) {
            cur_.pnl += pnl;
            cur_.cost += cost;
            if (position != 0.0 && !flipped) ++cur_.held;   // still held
        }
        if (was_open && (position == 0.0 || flipped)) {
            cur_.exit = row;
            cur_.closed = true;
            on_close(cur_);
            open_ = false;
        }
        if (position != 0.0 && !open_) {
            open_ = true;
            cur_ = RoundTrip{};
            cur_.entry = row;
            cur_.exit = row;
            cur_.side = position > 0.0 ? 1.0 : -1.0;
            cur_.pnl = was_open ? 0.0 : pnl;    // a flip's pnl went to the old trade
            cur_.cost = was_open ? 0.0 : cost;
            cur_.held = 1;
        }
        prev_pos_ = position;
    }

    bool open() const { return open_; }
    const RoundTrip& current() const { return cur_; }

private:
    bool open_ = false;
    double prev_pos_ = 0.0;
    RoundTrip cur_;
};

}} // namespace screamer::detail

#endif // SCREAMER_DETAIL_ROUND_TRIPS_H
