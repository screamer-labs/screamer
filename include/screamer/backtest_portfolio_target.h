#ifndef SCREAMER_BACKTEST_PORTFOLIO_TARGET_H
#define SCREAMER_BACKTEST_PORTFOLIO_TARGET_H

#include <cstddef>
#include <limits>
#include <string>
#include <vector>
#include "screamer/common/row_op.h"
#include "screamer/detail/portfolio_engine.h"

namespace screamer {

    // BacktestPortfolioTarget: K assets traded toward per-asset targets from
    // one capital account, filled at each step's price (like
    // BacktestPriceTarget, per asset). Inputs: target (T, K) and price (T, K).
    // Output: (T, K, 4), [equity, pnl, position, cost] per asset; the
    // portfolio is the sum over assets plus initial_capital. See
    // detail/portfolio_engine.h for the target, sizing, drift, and limit rules.
    class BacktestPortfolioTarget : public RowOp {
    public:
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
            : RowOp(2, 4),
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
    class BacktestPortfolioOHLCTarget : public RowOp {
    public:
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
            : RowOp(5, 4),
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
