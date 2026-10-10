#ifndef SCREAMER_BAND_SIGNAL_H
#define SCREAMER_BAND_SIGNAL_H

// BandSignal: a three-state mean-reversion position from a z-score.
//
//   flat:  z >= entry   -> -1 (short the spread, it is expected to fall)
//          z <= -entry  -> +1 (long the spread)
//   short: exits to 0 when z <= exit;  long: exits when z >= -exit
//   stop:  a short exits when z >= stop, a long when z <= -stop, and that
//          side is locked out until z returns inside the exit level, so a
//          spread that keeps diverging is not re-entered at once.
//
// A z-score that jumps from beyond one entry level to beyond the other in one
// step exits and enters the other side on that step. exit may be negative (hold
// a short until the spread overshoots below its mean) but must lie inside
// (-entry, entry). A NaN input is skipped: the output is NaN and the state is
// unchanged.
//
// O(1) per step.

#include <cmath>
#include <limits>
#include <stdexcept>
#include "screamer/common/base.h"
#include "screamer/common/float_info.h"

namespace screamer {

class BandSignal : public ScreamerBase {
public:
    BandSignal(double entry = 2.0, double exit = 0.0,
               double stop = std::numeric_limits<double>::infinity())
        : entry_(entry), exit_(exit), stop_(stop)
    {
        if (!(std::isfinite(entry_) && entry_ > 0.0))
            throw std::invalid_argument("entry must be finite and positive.");
        if (!(std::isfinite(exit_) && exit_ > -entry_ && exit_ < entry_))
            throw std::invalid_argument("exit must lie inside (-entry, entry).");
        if (!(stop_ > entry_))
            throw std::invalid_argument("stop must be larger than entry (inf for no stop).");
        reset();
    }

    void reset() override {
        pos_ = 0.0;
        lock_short_ = lock_long_ = false;
    }

private:
    double process_scalar(double z) override {
        if (isnan2(z)) return std::numeric_limits<double>::quiet_NaN();
        // Lockouts end once the z-score is back inside the exit level.
        if (lock_short_ && z <= exit_) lock_short_ = false;
        if (lock_long_ && z >= -exit_) lock_long_ = false;
        // Exits and stops.
        if (pos_ < 0.0) {
            if (z >= stop_) { pos_ = 0.0; lock_short_ = true; }
            else if (z <= exit_) pos_ = 0.0;
        } else if (pos_ > 0.0) {
            if (z <= -stop_) { pos_ = 0.0; lock_long_ = true; }
            else if (z >= -exit_) pos_ = 0.0;
        }
        // Entries, never beyond the stop.
        if (pos_ == 0.0) {
            if (z >= entry_ && z < stop_ && !lock_short_) pos_ = -1.0;
            else if (z <= -entry_ && z > -stop_ && !lock_long_) pos_ = 1.0;
        }
        return pos_;
    }

    const double entry_, exit_, stop_;
    double pos_ = 0.0;
    bool lock_short_ = false, lock_long_ = false;
};

}  // namespace screamer

#endif
