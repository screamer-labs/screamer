#ifndef SCREAMER_DETAIL_PORTFOLIO_ENGINE_H
#define SCREAMER_DETAIL_PORTFOLIO_ENGINE_H

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
#include "screamer/common/float_info.h"
#include "screamer/detail/pnl_account.h"
#include "screamer/detail/slippage.h"

namespace screamer { namespace detail {

    // PortfolioEngine: a target-position backtest of K assets that share one
    // capital account. The engines BacktestPortfolioTarget (prices) and
    // BacktestPortfolioOHLCTarget (OHLC bars) are thin RowOp wrappers on it.
    //
    // Targets. Each step gives one target per asset, in units, a currency
    // amount ("notional"), or a fraction of portfolio equity ("weight"). A
    // finite target is an order to trade the asset to that target on its
    // execution bar. NaN keeps the asset's position (and its last target, for
    // the drift check below), so a schedule rebalances by passing targets only
    // on rebalance steps.
    //
    // Equity. Portfolio equity is initial_capital plus the summed pnl of all
    // assets, marked at the execution reference prices; weights are fractions
    // of it, so returns compound across the whole book.
    //
    // Drift. With rebalance_band > 0 (notional and weight sizing), an asset
    // held toward a target is traded back to it when its holding drifts from
    // the target by more than rebalance_band times equity. rebalance="drifted"
    // trades only the assets beyond the band; "all" rebalances every asset
    // that has a target as soon as one is beyond it.
    //
    // Exposure limits. If the book after the orders would have gross exposure
    // above max_gross or absolute net exposure above max_net (both as
    // multiples of equity), every traded asset's target is scaled down by the
    // same factor so the book meets both limits.
    //
    // Execution. Each asset's order executes `delay` steps after its target at
    // the bar's `fill_at` price (the reference), moved by its slippage, and pays
    // its fee. An asset whose bar is NaN on a step does not trade or mark that
    // step: its output row repeats its equity with zero pnl and cost, so the
    // portfolio sum stays defined while it is not quoted. An order due on such
    // a step waits for the asset's next quoted bar, unless a newer order for
    // that asset is due first.
    //
    // Per-asset values (fee, slippage, tick_size, multiplier,
    // fee_per_contract) are given once for all assets or once per asset.
    class PortfolioEngine {
    public:
        PortfolioEngine(std::vector<double> fee, std::vector<double> slippage,
                        const std::string& slippage_unit, std::vector<double> tick_size,
                        std::vector<double> multiplier, std::vector<double> fee_per_contract,
                        const std::string& size, double initial_capital,
                        double rebalance_band, const std::string& rebalance,
                        double max_gross, double max_net,
                        const std::string& fill_at, int delay)
            : fee_(std::move(fee)), slip_amt_(std::move(slippage)), slip_unit_(slippage_unit),
              tick_(std::move(tick_size)), mult_(std::move(multiplier)),
              fpc_(std::move(fee_per_contract)), mode_(parse_size(size)),
              capital_(initial_capital), band_(rebalance_band),
              rebalance_all_(parse_rebalance(rebalance)),
              max_gross_(max_gross), max_net_(max_net),
              fill_at_(parse_fill_at(fill_at)), delay_(delay)
        {
            check_values(fee_, "fee", true);
            check_values(slip_amt_, "slippage", false);
            check_values(tick_, "tick_size", false);
            check_values(mult_, "multiplier", false);
            check_values(fpc_, "fee_per_contract", true);
            for (double m : mult_)
                if (!(m > 0.0)) throw std::invalid_argument("multiplier must be positive.");
            // Validates the slippage unit and amounts (ticks need a tick size).
            for (std::size_t i = 0; i < std::max(slip_amt_.size(), tick_.size()); ++i)
                Slippage(pick(slip_amt_, i), slip_unit_, pick(tick_, i));
            if (!std::isfinite(capital_) || capital_ < 0.0)
                throw std::invalid_argument("initial_capital must be finite and non-negative.");
            if (mode_ == Mode::Weight && !(capital_ > 0.0))
                throw std::invalid_argument("size=\"weight\" needs a positive initial_capital.");
            if (!std::isfinite(band_) || band_ < 0.0)
                throw std::invalid_argument("rebalance_band must be finite and non-negative.");
            if (band_ > 0.0 && mode_ == Mode::Units)
                throw std::invalid_argument("rebalance_band needs size=\"notional\" or \"weight\".");
            if (!(max_gross_ > 0.0) || !(max_net_ >= 0.0) || isnan2(max_gross_))
                throw std::invalid_argument("max_gross must be positive and max_net non-negative.");
            if ((std::isfinite(max_gross_) || std::isfinite(max_net_)) && !(capital_ > 0.0))
                throw std::invalid_argument("max_gross and max_net need a positive initial_capital.");
            if (delay_ < 0) throw std::invalid_argument("delay must be non-negative.");
            if (delay_ == 0 && fill_at_ != FillAt::Close)
                throw std::invalid_argument(
                    "delay=0 requires fill_at=\"close\": a target decided at a bar's "
                    "close cannot trade at an earlier price of that bar.");
        }

        void reset() {
            k_ = 0;
            accts_.clear();
            slips_.clear();
            working_.clear();
            queue_.clear();
            pending_.clear();
        }

        // Fix the number of assets and build the per-asset state.
        void set_width(std::size_t k) {
            for (const auto* v : {&fee_, &slip_amt_, &tick_, &mult_, &fpc_})
                if (v->size() != 1 && v->size() != k)
                    throw std::invalid_argument(
                        "per-asset values must have one entry or one per asset.");
            k_ = k;
            accts_.clear();
            slips_.clear();
            for (std::size_t i = 0; i < k; ++i) {
                accts_.emplace_back(pick(mult_, i), pick(fpc_, i));
                slips_.emplace_back(pick(slip_amt_, i), slip_unit_, pick(tick_, i));
            }
            working_.assign(k, nan());
            queue_.assign(static_cast<std::size_t>(delay_) * k, nan());
            head_ = 0;
            due_.assign(k, nan());
            pending_.assign(k, nan());
            ref_.assign(k, nan());
            goal_.assign(k, 0.0);
            trade_.assign(k, 0);
        }

        // One step. target, open, high, low, close each point at k values;
        // out receives k rows of [equity, pnl, position, cost].
        void step(const double* target, const double* open, const double* high,
                  const double* low, const double* close, double* out) {
            const std::size_t k = k_;
            // 1. The due targets: this step's for delay 0, else from the queue.
            if (delay_ == 0) {
                std::copy_n(target, k, due_.begin());
            } else {
                double* slot = queue_.data() + head_ * k;
                std::copy_n(slot, k, due_.begin());
                std::copy_n(target, k, slot);
                head_ = (head_ + 1) % static_cast<std::size_t>(delay_);
            }
            // An order that could not execute on an unquoted bar is still due.
            for (std::size_t i = 0; i < k; ++i) {
                if (isnan2(due_[i])) due_[i] = pending_[i];
                pending_[i] = nan();
            }
            // 2. Reference prices; equity marked at them.
            double equity = capital_;
            for (std::size_t i = 0; i < k; ++i) {
                const bool quoted = !(isnan2(open[i]) || isnan2(high[i]) ||
                                      isnan2(low[i]) || isnan2(close[i]));
                ref_[i] = quoted ? reference(open[i], high[i], low[i], close[i]) : nan();
                equity += quoted ? accts_[i].equity_at(ref_[i]) : accts_[i].equity();
            }
            // 3. Orders: finite due targets, then drifted holdings.
            bool drift = false;
            for (std::size_t i = 0; i < k; ++i) {
                goal_[i] = accts_[i].position();
                trade_[i] = 0;
                if (isnan2(ref_[i])) {
                    pending_[i] = due_[i];
                    continue;
                }
                if (!isnan2(due_[i])) {
                    working_[i] = due_[i];
                    trade_[i] = 1;
                } else if (band_ > 0.0 && !isnan2(working_[i]) && drifted(i, equity)) {
                    trade_[i] = 2;
                    drift = true;
                }
            }
            if (drift && rebalance_all_)
                for (std::size_t i = 0; i < k; ++i)
                    if (!trade_[i] && !isnan2(ref_[i]) && !isnan2(working_[i])) trade_[i] = 2;
            for (std::size_t i = 0; i < k; ++i)
                if (trade_[i]) goal_[i] = units(working_[i], ref_[i], equity, i);
            // 4. Exposure limits on the book after the orders.
            if (std::isfinite(max_gross_) || std::isfinite(max_net_)) limit(equity);
            // 5. Execute and mark.
            for (std::size_t i = 0; i < k; ++i) {
                double* row = out + 4 * i;
                PnLAccount& a = accts_[i];
                if (isnan2(ref_[i])) {
                    row[0] = a.equity(); row[1] = 0.0; row[2] = a.position(); row[3] = 0.0;
                    continue;
                }
                const double dpos = trade_[i] ? goal_[i] - a.position() : 0.0;
                const double fill = dpos != 0.0 ? slips_[i].fill(ref_[i], dpos > 0.0) : ref_[i];
                auto [e, p, pos, c] = a.step(close[i], dpos, fill, pick(fee_, i),
                                             pick(fpc_, i), ref_[i]);
                row[0] = e; row[1] = p; row[2] = pos; row[3] = c;
            }
        }

    private:
        enum class Mode { Units, Notional, Weight };
        enum class FillAt { Open, Close, Typical, OHLC4 };

        static double nan() { return std::numeric_limits<double>::quiet_NaN(); }

        static double pick(const std::vector<double>& v, std::size_t i) {
            return v.size() == 1 ? v[0] : v[i];
        }

        static void check_values(const std::vector<double>& v, const char* name, bool signed_ok) {
            if (v.empty())
                throw std::invalid_argument(std::string(name) + " needs at least one value.");
            for (double x : v)
                if (!std::isfinite(x) || (!signed_ok && x < 0.0))
                    throw std::invalid_argument(std::string(name) +
                        (signed_ok ? " must be finite." : " must be finite and non-negative."));
        }

        static Mode parse_size(const std::string& s) {
            if (s == "units") return Mode::Units;
            if (s == "notional") return Mode::Notional;
            if (s == "weight") return Mode::Weight;
            throw std::invalid_argument("size must be \"units\", \"notional\", or \"weight\".");
        }

        static bool parse_rebalance(const std::string& s) {
            if (s == "drifted") return false;
            if (s == "all") return true;
            throw std::invalid_argument("rebalance must be \"drifted\" or \"all\".");
        }

        static FillAt parse_fill_at(const std::string& f) {
            if (f == "open") return FillAt::Open;
            if (f == "close") return FillAt::Close;
            if (f == "typical") return FillAt::Typical;
            if (f == "ohlc4") return FillAt::OHLC4;
            throw std::invalid_argument(
                "fill_at must be \"open\", \"close\", \"typical\", or \"ohlc4\".");
        }

        double reference(double o, double h, double l, double c) const {
            switch (fill_at_) {
                case FillAt::Open:    return o;
                case FillAt::Close:   return c;
                case FillAt::Typical: return (h + l + c) / 3.0;
                case FillAt::OHLC4:   return (o + h + l + c) / 4.0;
            }
            return c;
        }

        // Units for a target value at price `ref` with portfolio equity `equity`.
        double units(double value, double ref, double equity, std::size_t i) const {
            const double per_unit = ref * pick(mult_, i);
            switch (mode_) {
                case Mode::Units:    return value;
                case Mode::Notional: return per_unit != 0.0 ? value / per_unit : 0.0;
                case Mode::Weight:   return equity > 0.0 && per_unit != 0.0
                                         ? value * equity / per_unit : 0.0;
            }
            return value;
        }

        bool drifted(std::size_t i, double equity) const {
            const double held = accts_[i].position() * ref_[i] * pick(mult_, i);
            const double wanted = mode_ == Mode::Weight ? working_[i] * equity : working_[i];
            return std::abs(held - wanted) > band_ * std::abs(equity);
        }

        // Scale the traded assets' goals so the book meets max_gross and
        // max_net. Assets that do not trade this step keep their position.
        void limit(double equity) {
            if (!(equity > 0.0)) return;
            double gross = 0.0, net = 0.0, tg = 0.0, tn = 0.0;
            for (std::size_t i = 0; i < k_; ++i) {
                const double px = isnan2(ref_[i]) ? 0.0 : ref_[i];
                const double v = goal_[i] * px * pick(mult_, i);
                gross += std::abs(v); net += v;
                if (trade_[i]) { tg += std::abs(v); tn += v; }
            }
            gross /= equity; net /= equity; tg /= equity; tn /= equity;
            double s = 1.0;
            if (gross > max_gross_ && tg > 0.0)
                s = std::min(s, std::max(0.0, (max_gross_ - (gross - tg)) / tg));
            if (std::abs(net) > max_net_ && tn != 0.0) {
                // Solve |(net - tn) + s * tn| <= max_net for the largest s <= 1.
                const double rest = net - tn;
                const double target = net > 0.0 ? max_net_ : -max_net_;
                s = std::min(s, std::max(0.0, (target - rest) / tn));
            }
            if (s < 1.0)
                for (std::size_t i = 0; i < k_; ++i)
                    if (trade_[i]) goal_[i] *= s;
        }

        std::vector<double> fee_, slip_amt_;
        std::string slip_unit_;
        std::vector<double> tick_, mult_, fpc_;
        Mode mode_;
        double capital_, band_;
        bool rebalance_all_;
        double max_gross_, max_net_;
        FillAt fill_at_;
        int delay_;

        std::size_t k_ = 0;
        std::vector<PnLAccount> accts_;
        std::vector<Slippage> slips_;
        std::vector<double> working_, queue_, due_, pending_, ref_, goal_;
        std::vector<char> trade_;
        std::size_t head_ = 0;
    };

}} // namespace screamer::detail

#endif // SCREAMER_DETAIL_PORTFOLIO_ENGINE_H
