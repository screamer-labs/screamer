#ifndef SCREAMER_BACKTEST_TRADES_ORDERS_H
#define SCREAMER_BACKTEST_TRADES_ORDERS_H

#include <limits>
#include <stdexcept>
#include <string>
#include "screamer/common/functor_base.h"
#include "screamer/detail/trades_fill.h"

namespace screamer {

    // BacktestTradesOrders: 6 -> 4. Post a two-sided quote on the trade tape;
    // a resting bid fills as a maker at its price when a sell-print crosses it
    // (touch/breach). An arriving bid (not quoted on the previous event,
    // re-priced, or its previous order fully filled) above the print, and a
    // market bid (market_limit returns +inf), fill as a taker at the print,
    // capped at the bid. The sell side is symmetric. Inventory is capped to
    // [min_position, max_position]. Outputs [equity, pnl, position, cost].
    class BacktestTradesOrders : public FunctorBase<BacktestTradesOrders, 6, 4> {
    public:
        static constexpr const char* kTypeName = "BacktestTradesOrders";
        static constexpr int kStateVersion = 1;

        template <class Ar>
        void settings(Ar& ar) {
            ar("maker_fee", core_.maker_fee());
            ar("taker_fee", core_.taker_fee());
            ar("fill", std::string(core_.breach() ? "breach" : "touch"));
            ar("participation_ratio", core_.participation());
            ar("tick_size", core_.slippage().tick_size());
            ar("min_position", core_.min_position());
            ar("max_position", core_.max_position());
            ar("multiplier", core_.multiplier());
            ar("maker_fee_per_contract", core_.maker_fee_per_contract());
            ar("taker_fee_per_contract", core_.taker_fee_per_contract());
            ar("slippage", core_.slippage().amount());
            ar("slippage_unit", std::string(core_.slippage().unit_name()));
        }

        template <class Ar>
        void state(Ar& ar) { core_.state(ar); }

        BacktestTradesOrders(double maker_fee = 0.0, double taker_fee = 0.0,
                             const std::string& fill = "touch",
                             double participation_ratio = 1.0, double tick_size = 0.0,
                             double min_position = -std::numeric_limits<double>::infinity(),
                             double max_position = std::numeric_limits<double>::infinity(),
                           double multiplier = 1.0,
                           double maker_fee_per_contract = 0.0,
                           double taker_fee_per_contract = 0.0,
                           double slippage = 0.0,
                           const std::string& slippage_unit = "fraction")
            : core_(maker_fee, taker_fee, parse_fill(fill),
                    parse_participation(participation_ratio),
                    detail::Slippage(slippage, slippage_unit, tick_size),
                    min_position, max_position,
                    multiplier, maker_fee_per_contract, taker_fee_per_contract) {}
        void reset() override { core_.reset(); }
        ResultTuple call(const InputArray& in) override {
            return core_.quote(in[0], in[1], in[2], in[3], in[4], in[5]);
        }
    private:
        static bool parse_fill(const std::string& f) {
            if (f == "touch") return false;
            if (f == "breach") return true;
            throw std::invalid_argument("fill must be \"touch\" or \"breach\".");
        }
        static double parse_participation(double p) {
            if (!(p > 0.0) || p > 1.0)
                throw std::invalid_argument("participation_ratio must be in (0, 1].");
            return p;
        }
        detail::TradesFill core_;
    };

} // namespace screamer

#endif // SCREAMER_BACKTEST_TRADES_ORDERS_H
