#ifndef SCREAMER_BACKTEST_PORTFOLIO_TARGET_H
#define SCREAMER_BACKTEST_PORTFOLIO_TARGET_H

#include <cstddef>
#include <limits>
#include <string>
#include <vector>
#include "screamer/common/row_op.h"
#include "screamer/detail/portfolio_engine.h"

namespace screamer {

    namespace detail {
        // A per-asset setting: a number when one value serves every asset.
        template <class Ar>
        void per_asset(Ar& ar, const char* name, const std::vector<double>& v) {
            if (v.size() == 1) ar(name, v[0]);
            else ar(name, v);
        }
    } // namespace detail

    // BacktestPortfolioTarget: K assets traded toward per-asset targets from
    // one capital account, filled at each step's price (like
    // BacktestPriceTarget, per asset). Inputs: target (T, K) and price (T, K).
    // Output: (T, K, 4), [equity, pnl, position, cost] per asset; the
    // portfolio is the sum over assets plus initial_capital. See
    // detail/portfolio_engine.h for the target, sizing, drift, and limit rules.
    class BacktestPortfolioTarget : public serial::Saveable<BacktestPortfolioTarget, RowOp> {
    public:
        static constexpr const char* kTypeName = "BacktestPortfolioTarget";
        static constexpr int kStateVersion = 1;

        template <class Ar>
        void settings(Ar& ar) {
            const auto& e = engine_;
            detail::per_asset(ar, "fee", e.fee());
            detail::per_asset(ar, "slippage", e.slippage());
            ar("slippage_unit", e.slippage_unit());
            detail::per_asset(ar, "tick_size", e.tick_size());
            detail::per_asset(ar, "multiplier", e.multiplier());
            detail::per_asset(ar, "fee_per_contract", e.fee_per_contract());
            ar("size", std::string(e.size_name()));
            ar("initial_capital", e.initial_capital());
            ar("rebalance_band", e.rebalance_band());
            ar("rebalance", std::string(e.rebalance_name()));
            ar("max_gross", e.max_gross());
            ar("max_net", e.max_net());
        }

        template <class Ar>
        void state(Ar& ar) {
            width_state(ar);
            engine_.state(ar, width());
        }

        // The market input (after the target) that marks a position: the price.
        // walk_forward's carry marks with it.
        static constexpr std::size_t mark_input = 0;

        BacktestPortfolioTarget(std::vector<double> fee = {0.0},
                                std::vector<double> slippage = {0.0},
                                const std::string& slippage_unit = "fraction",
                                std::vector<double> tick_size = {0.0},
                                std::vector<double> multiplier = {1.0},
                                std::vector<double> fee_per_contract = {0.0},
                                const std::string& size = "weight",
                                double initial_capital = 1.0,
                                double rebalance_band = 0.0,
                                const std::string& rebalance = "drifted",
                                double max_gross = std::numeric_limits<double>::infinity(),
                                double max_net = std::numeric_limits<double>::infinity())
            : serial::Saveable<BacktestPortfolioTarget, RowOp>(2, 4),
              engine_(std::move(fee), std::move(slippage), slippage_unit, std::move(tick_size),
                      std::move(multiplier), std::move(fee_per_contract), size,
                      initial_capital, rebalance_band, rebalance, max_gross, max_net,
                      "close", 0) {}

    protected:
        void on_width(std::size_t k) override { engine_.set_width(k); }
        void reset_state() override { engine_.reset(); }
        void map_row(const double* in, double* out, std::size_t k) override {
            const double* price = in + k;
            engine_.step(in, price, price, price, price, out);
        }

    private:
        detail::PortfolioEngine engine_;
    };

    // BacktestPortfolioOHLCTarget: the portfolio engine on OHLC bars. Inputs:
    // target, open, high, low, close, each (T, K). The target decided on bar t
    // executes `delay` bars later (default 1) at the bar's `fill_at` price
    // ("open", "close", "typical", "ohlc4"); delay = 0 needs fill_at = "close".
    class BacktestPortfolioOHLCTarget : public serial::Saveable<BacktestPortfolioOHLCTarget, RowOp> {
    public:
        static constexpr const char* kTypeName = "BacktestPortfolioOHLCTarget";
        static constexpr int kStateVersion = 1;

        template <class Ar>
        void settings(Ar& ar) {
            const auto& e = engine_;
            detail::per_asset(ar, "fee", e.fee());
            detail::per_asset(ar, "slippage", e.slippage());
            ar("slippage_unit", e.slippage_unit());
            detail::per_asset(ar, "tick_size", e.tick_size());
            detail::per_asset(ar, "multiplier", e.multiplier());
            detail::per_asset(ar, "fee_per_contract", e.fee_per_contract());
            ar("size", std::string(e.size_name()));
            ar("initial_capital", e.initial_capital());
            ar("rebalance_band", e.rebalance_band());
            ar("rebalance", std::string(e.rebalance_name()));
            ar("max_gross", e.max_gross());
            ar("max_net", e.max_net());
            ar("fill_at", std::string(engine_.fill_at_name()));
            ar("delay", engine_.delay());
        }

        template <class Ar>
        void state(Ar& ar) {
            width_state(ar);
            engine_.state(ar, width());
        }

        // The market input (after the target) that marks a position: the close.
        // walk_forward's carry marks with it.
        static constexpr std::size_t mark_input = 3;

        BacktestPortfolioOHLCTarget(std::vector<double> fee = {0.0},
                                    std::vector<double> slippage = {0.0},
                                    const std::string& slippage_unit = "fraction",
                                    std::vector<double> tick_size = {0.0},
                                    std::vector<double> multiplier = {1.0},
                                    std::vector<double> fee_per_contract = {0.0},
                                    const std::string& size = "weight",
                                    double initial_capital = 1.0,
                                    double rebalance_band = 0.0,
                                    const std::string& rebalance = "drifted",
                                    double max_gross = std::numeric_limits<double>::infinity(),
                                    double max_net = std::numeric_limits<double>::infinity(),
                                    const std::string& fill_at = "open",
                                    int delay = 1)
            : serial::Saveable<BacktestPortfolioOHLCTarget, RowOp>(5, 4),
              engine_(std::move(fee), std::move(slippage), slippage_unit, std::move(tick_size),
                      std::move(multiplier), std::move(fee_per_contract), size,
                      initial_capital, rebalance_band, rebalance, max_gross, max_net,
                      fill_at, delay) {}

    protected:
        void on_width(std::size_t k) override { engine_.set_width(k); }
        void reset_state() override { engine_.reset(); }
        void map_row(const double* in, double* out, std::size_t k) override {
            engine_.step(in, in + k, in + 2 * k, in + 3 * k, in + 4 * k, out);
        }

    private:
        detail::PortfolioEngine engine_;
    };

} // namespace screamer

#endif // SCREAMER_BACKTEST_PORTFOLIO_TARGET_H
