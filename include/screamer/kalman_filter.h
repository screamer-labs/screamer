#ifndef SCREAMER_KALMAN_FILTER_H
#define SCREAMER_KALMAN_FILTER_H

// KalmanFilter: scalar 1-D Kalman filter with constant unit
// transition and observation matrices. The classic "noisy random
// walk" model:
//
//   prediction:  x_pred  = x_prev
//                P_pred  = P_prev + process_var
//   update:      K       = P_pred / (P_pred + observation_var)
//                x       = x_pred + K * (z - x_pred)
//                P       = (1 - K) * P_pred
//
// where z is the new observation, P is the running variance estimate
// of the state, and K is the Kalman gain. After many samples K
// approaches a constant determined by the ratio process_var / observation_var,
// at which point the filter is an exponential smoother with that
// effective alpha.
//
// 1 -> 1, O(1) per step. Initial state = 0, initial variance = 1.0
// by default; both are optional constructor parameters.
//
// Robust update (huber, default inf = off): with innovation e = z - x_pred
// and its predicted variance Q = P_pred + observation_var, let
// |z_std| = |e| / sqrt(Q). When |z_std| > huber the update uses the
// observation variance observation_var / w, w = huber / |z_std|, in place of
// observation_var (a Huber M-estimate step, Masreliez and Martin 1977). An
// outlier's pull on the state is then bounded, however large it is; smaller
// innovations update exactly as before.

#include "screamer/common/finite_settings.h"
#include <cmath>
#include <stdexcept>
#include <limits>
#include "screamer/common/base.h"
#include "screamer/common/float_info.h"

namespace screamer {

class KalmanFilter : public ScreamerBase {
public:
    KalmanFilter(double process_var,
                 double observation_var,
                 double initial_state = 0.0,
                 double initial_variance = 1.0,
                 double huber = std::numeric_limits<double>::infinity())
        : process_var_(process_var),
          observation_var_(observation_var),
          initial_state_(initial_state),
          initial_variance_(initial_variance),
          huber_(huber),
          robust_(huber < std::numeric_limits<double>::infinity())
    {
        require_finite("process_var", process_var);
        require_finite("observation_var", observation_var);
        require_finite("initial_state", initial_state);
        require_finite("initial_variance", initial_variance, kTakesPosInf);   // inf: a diffuse prior
        require_finite("huber", huber, kTakesPosInf);                         // inf: off
        if (!(process_var >= 0.0)) {
            throw std::invalid_argument("process_var must be >= 0.");
        }
        if (!(observation_var > 0.0)) {
            throw std::invalid_argument("observation_var must be > 0.");
        }
        if (!(initial_variance >= 0.0)) {
            throw std::invalid_argument("initial_variance must be >= 0.");
        }
        if (!(huber > 0.0)) {
            throw std::invalid_argument("huber must be > 0.");
        }
        reset();
    }

    void reset() override {
        x_ = initial_state_;
        P_ = initial_variance_;
        diffuse_pending_ = std::isinf(initial_variance_);
    }

    double process_scalar(double z) override {
        // NaN policy "ignore": missing observation -> emit NaN, do not
        // run the predict/update steps so the state stays exactly where
        // it was. See docs/nan_policy.md.
        if (isnan2(z)) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        if (diffuse_pending_) {
            // A diffuse prior (initial_variance = inf), decided by the
            // setting: the first observation is the limit of the update as
            // P -> inf. The gain is 1, the state is the observation and its
            // variance the observation variance. A finite variance never
            // takes this path, so it gives the 2.8.0 results bit for bit
            // (NaN where its arithmetic overflows).
            x_ = z;
            P_ = observation_var_;
            diffuse_pending_ = false;
            return x_;
        }
        // Predict.
        const double P_pred = P_ + process_var_;
        const double e = z - x_;
        double R = observation_var_;
        if (robust_) {
            // Huber weight on the standardized innovation: inflate the
            // observation variance of an outlier to R / w.
            const double az = std::fabs(e) / std::sqrt(P_pred + R);
            if (az > huber_) R /= huber_ / az;
        }
        // Update.
        const double K = P_pred / (P_pred + R);
        x_ = x_ + K * e;
        P_ = (1.0 - K) * P_pred;
        return x_;
    }

private:
    const double process_var_;
    const double observation_var_;
    const double initial_state_;
    const double initial_variance_;
    const double huber_;
    const bool robust_;              // huber finite
    double x_ = 0.0;
    double P_ = 0.0;
    bool diffuse_pending_ = false;   // inf initial_variance, before the first observation
};

}  // namespace screamer

#endif
