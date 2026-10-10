#ifndef SCREAMER_DETAIL_SIZER_H
#define SCREAMER_DETAIL_SIZER_H

#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include "screamer/common/float_info.h"

namespace screamer { namespace detail {

    // Sizer: turns a target into a position in units for the target engines.
    //   "units"     the target is the position itself (no conversion)
    //   "notional"  the target is a currency amount: units = target / (price * multiplier)
    //   "weight"    the target is a fraction of equity:
    //               units = target * (initial_capital + equity) / (price * multiplier)
    // `price` is the execution reference price (before slippage) and `equity`
    // the engine's cumulative pnl marked at that price, so weights compound.
    //
    // The units are fixed when the target value changes and kept while it stays
    // the same, so a constant weight does not trade on every price move. With
    // rebalance_band > 0 the units are recomputed when the held amount drifts
    // from the target amount by more than rebalance_band * (initial_capital +
    // equity); for weights that is a drift of rebalance_band in weight.
    class Sizer {
    public:
        Sizer(const std::string& size = "units", double initial_capital = 0.0,
              double rebalance_band = 0.0, double multiplier = 1.0)
            : mode_(parse(size)), capital_(initial_capital), band_(rebalance_band),
              multiplier_(multiplier)
        {
            if (!std::isfinite(capital_) || capital_ < 0.0)
                throw std::invalid_argument("initial_capital must be finite and non-negative.");
            if (!std::isfinite(band_) || band_ < 0.0)
                throw std::invalid_argument("rebalance_band must be finite and non-negative.");
            if (mode_ == Mode::Units && band_ > 0.0)
                throw std::invalid_argument("rebalance_band needs size=\"notional\" or \"weight\".");
            if ((mode_ == Mode::Weight || band_ > 0.0) && !(capital_ > 0.0))
                throw std::invalid_argument(
                    "size=\"weight\" and rebalance_band need a positive initial_capital.");
            reset();
        }

        bool active() const { return mode_ != Mode::Units; }

        void reset() {
            has_ = false;
            value_ = units_ = std::numeric_limits<double>::quiet_NaN();
        }

        // Units for target `value`; NaN (hold) when the value or the price
        // cannot be used.
        double units(double value, double price, double position, double equity) {
            if (mode_ == Mode::Units || isnan2(value)) return value;
            if (!(price > 0.0) || !std::isfinite(price))
                return std::numeric_limits<double>::quiet_NaN();
            const double basis = capital_ + equity;
            if (!has_ || value != value_) {
                units_ = convert(value, price, basis);
                value_ = value;
                has_ = true;
            } else if (band_ > 0.0) {
                const double held = position * price * multiplier_;
                const double wanted = mode_ == Mode::Notional ? value : value * basis;
                if (std::abs(held - wanted) > band_ * std::abs(basis))
                    units_ = convert(value, price, basis);
            }
            return units_;
        }

    private:
        enum class Mode { Units, Notional, Weight };

        static Mode parse(const std::string& s) {
            if (s == "units") return Mode::Units;
            if (s == "notional") return Mode::Notional;
            if (s == "weight") return Mode::Weight;
            throw std::invalid_argument("size must be \"units\", \"notional\", or \"weight\".");
        }

        double convert(double value, double price, double basis) const {
            const double per_unit = price * multiplier_;
            if (mode_ == Mode::Notional) return value / per_unit;
            return basis > 0.0 ? value * basis / per_unit : 0.0;   // no equity left: flat
        }

        Mode mode_;
        double capital_, band_, multiplier_;
        bool has_ = false;
        double value_ = std::numeric_limits<double>::quiet_NaN();
        double units_ = std::numeric_limits<double>::quiet_NaN();
    };

}} // namespace screamer::detail

#endif // SCREAMER_DETAIL_SIZER_H
