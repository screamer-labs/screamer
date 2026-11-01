#ifndef SCREAMER_BACKTEST_OHLC_ORDERS_H
#define SCREAMER_BACKTEST_OHLC_ORDERS_H

#include <limits>
#include <stdexcept>
#include <string>
#include "screamer/common/functor_base.h"
#include "screamer/detail/ohlc_fill.h"

namespace screamer {

    // BacktestOHLCOrders: 8 -> 4. Post a two-sided quote on bars, placed at
    // each bar's open. A bid above the open (an ask below it) or a market order
    // fills as a taker at the open moved by the slippage model, capped at the
    // limit; otherwise a side fills as a maker at its price when the bar's
    // low/high reaches it (touch/breach). Inventory is capped to
    // [min_position, max_position].
    // Outputs [equity, pnl, position, cost].
    class BacktestOHLCOrders : public FunctorBase<BacktestOHLCOrders, 8, 4> {
    public:
        static constexpr const char* kTypeName = "BacktestOHLCOrders";
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

        BacktestOHLCOrders(double maker_fee = 0.0, double taker_fee = 0.0,
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
            return core_.quote(in[0], in[1], in[2], in[3], in[4], in[5], in[6], in[7]);
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
        detail::OHLCFill core_;
    };

} // namespace screamer

#endif // SCREAMER_BACKTEST_OHLC_ORDERS_H
