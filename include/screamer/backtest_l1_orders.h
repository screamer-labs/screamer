#ifndef SCREAMER_BACKTEST_L1_ORDERS_H
#define SCREAMER_BACKTEST_L1_ORDERS_H

#include <limits>
#include <stdexcept>
#include <string>
#include "screamer/common/functor_base.h"
#include "screamer/detail/l1_fill.h"

namespace screamer {

    // BacktestL1Orders: 8 -> 4. Post a two-sided quote against an L1 book.
    // Inputs (bid_price, bid_size, ask_price, ask_size,
    //         market_bid, market_ask, market_bid_size, market_ask_size).
    // A resting bid (quoted at the same price on the previous event) fills as a
    // maker when the market ask trades through it (breach) or locks it in touch
    // mode. A bid arrives when its side was not quoted on the previous event,
    // its price changed, or its previous order was fully filled; an arriving bid
    // at or above the market ask is a taker from the touch, capped at its limit.
    // The sell side is symmetric. Inventory capped to [min_position,
    // max_position]. Outputs [equity, pnl, position, cost].
    // nan_policy: ignore on the market quote; a NaN own-quote price means
    // that side is not quoted.
    class BacktestL1Orders : public FunctorBase<BacktestL1Orders, 8, 4> {
    public:
        static constexpr const char* kTypeName = "BacktestL1Orders";
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

        BacktestL1Orders(double maker_fee = 0.0, double taker_fee = 0.0,
                         const std::string& fill = "breach",
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
            return core_.quote(in[0], in[1], in[2], in[3],
                               in[4], in[5], in[6], in[7]);
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

        detail::L1Fill core_;
    };

} // namespace screamer

#endif // SCREAMER_BACKTEST_L1_ORDERS_H
