#ifndef SCREAMER_DETAIL_SLIPPAGE_H
#define SCREAMER_DETAIL_SLIPPAGE_H

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace screamer { namespace detail {

    // Slippage: the adverse price offset applied to a taker (market) fill, in
    // whole ticks of the instrument's price grid. One amount is read in one unit:
    //   "fraction": offset = reference * amount   (5 bps = 5e-4)
    //   "ticks":    offset = amount * tick_size   (needs tick_size > 0)
    //   "price":    offset = amount               (absolute, in price units)
    // A buy fills at reference + offset, a sell at reference - offset. When
    // tick_size > 0 the offset is rounded up to a whole number of ticks, so an
    // on-grid reference gives an on-grid fill and an average fill price (a VWAP
    // across book levels) keeps its average. Zero slippage leaves the price as is.
    // Resting limit fills never use this: they fill at their own limit price.
    class Slippage {
    public:
        Slippage(double amount = 0.0, const std::string& unit = "fraction",
                 double tick_size = 0.0)
            : amount_(amount), unit_(parse_unit(unit)), tick_(tick_size)
        {
            if (!std::isfinite(amount_) || amount_ < 0.0)
                throw std::invalid_argument("slippage must be finite and non-negative.");
            if (!std::isfinite(tick_) || tick_ < 0.0)
                throw std::invalid_argument("tick_size must be finite and non-negative.");
            if (unit_ == Unit::Ticks && amount_ > 0.0 && !(tick_ > 0.0))
                throw std::invalid_argument(
                    "slippage_unit=\"ticks\" needs a positive tick_size.");
        }

        double tick_size() const { return tick_; }

        double buy(double reference) const { return reference + offset(reference); }

        double sell(double reference) const { return reference - offset(reference); }

        double fill(double reference, bool is_buy) const {
            return is_buy ? buy(reference) : sell(reference);
        }

    private:
        enum class Unit { Fraction, Ticks, Price };

        static Unit parse_unit(const std::string& u) {
            if (u == "fraction") return Unit::Fraction;
            if (u == "ticks") return Unit::Ticks;
            if (u == "price") return Unit::Price;
            throw std::invalid_argument(
                "slippage_unit must be \"fraction\", \"ticks\", or \"price\".");
        }

        double offset(double reference) const {
            double raw = 0.0;
            switch (unit_) {
                case Unit::Fraction: raw = std::abs(reference) * amount_; break;
                case Unit::Ticks:    raw = amount_ * tick_; break;
                case Unit::Price:    raw = amount_; break;
            }
            return round_up_to_ticks(raw);
        }

        // Whole ticks, rounded up, with a small tolerance so an offset that is
        // a whole number of ticks up to floating-point error is not pushed a
        // whole tick further.
        double round_up_to_ticks(double offset) const {
            if (!(tick_ > 0.0) || !(offset > 0.0)) return offset;
            const double q = offset / tick_;
            const double r = std::round(q);
            return (std::abs(q - r) < 1e-9 ? r : std::ceil(q)) * tick_;
        }

        double amount_;
        Unit unit_;
        double tick_;
    };

    // Fill price of a marketable order of `remaining` units at `limit` against
    // the opposite touch (opp_price, opp_size): the displayed size at the touch,
    // the rest one tick further, the average then moved by the slippage model.
    // Neither the walk nor the slippage passes the limit; a +/-inf limit (market
    // order) leaves both uncapped. remaining > 0.
    inline double marketable_fill(bool buy, double limit, double remaining,
                                  double opp_price, double opp_size,
                                  const Slippage& slippage) {
        const double disp = std::min(remaining, opp_size);
        const double over = remaining - disp;
        const double tick = slippage.tick_size();
        const double walk = buy ? std::min(opp_price + tick, limit)
                                : std::max(opp_price - tick, limit);
        const double fill = slippage.fill((disp * opp_price + over * walk) / remaining, buy);
        return buy ? std::min(fill, limit) : std::max(fill, limit);
    }

}} // namespace screamer::detail

#endif // SCREAMER_DETAIL_SLIPPAGE_H
