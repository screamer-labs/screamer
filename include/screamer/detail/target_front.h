#ifndef SCREAMER_DETAIL_TARGET_FRONT_H
#define SCREAMER_DETAIL_TARGET_FRONT_H

#include <algorithm>
#include <limits>
#include <tuple>

namespace screamer { namespace detail {

    // Convert a desired target position into a marketable one-sided order that
    // moves the live position toward clamp(target, [min_pos, max_pos]). Returns a
    // canonical two-sided quote (bid_price, bid_size, ask_price, ask_size) with a
    // market price (+inf buy / -inf sell) on the trading side and zero size on the
    // idle side, so any fill core's marketable path executes it. The core still caps
    // the fill to the room, so a target beyond the cap lands exactly on the cap.
    inline std::tuple<double, double, double, double>
    target_to_quote(double target, double position, double min_pos, double max_pos) {
        const double clamped = std::clamp(target, min_pos, max_pos);
        const double delta = clamped - position;
        const double inf = std::numeric_limits<double>::infinity();
        if (delta > 0.0) return std::make_tuple(inf, delta, -inf, 0.0);   // market buy
        if (delta < 0.0) return std::make_tuple(inf, 0.0, -inf, -delta);  // market sell
        return std::make_tuple(inf, 0.0, -inf, 0.0);                      // no trade
    }

}} // namespace screamer::detail

#endif // SCREAMER_DETAIL_TARGET_FRONT_H
