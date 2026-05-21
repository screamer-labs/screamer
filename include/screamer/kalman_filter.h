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
                 double initial_variance = 1.0)
        : process_var_(process_var),
          observation_var_(observation_var),
          initial_state_(initial_state),
          initial_variance_(initial_variance)
    {
        if (!(process_var >= 0.0)) {
            throw std::invalid_argument("process_var must be >= 0.");
        }
        if (!(observation_var > 0.0)) {
            throw std::invalid_argument("observation_var must be > 0.");
        }
        if (!(initial_variance >= 0.0)) {
            throw std::invalid_argument("initial_variance must be >= 0.");
        }
        reset();
    }

    void reset() override {
        x_ = initial_state_;
        P_ = initial_variance_;
    }

    double process_scalar(double z) override {
        // NaN policy "ignore": missing observation -> emit NaN, do not
        // run the predict/update steps so the state stays exactly where
        // it was. See docs/nan_policy.md.
        if (isnan2(z)) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        // Predict.
        const double P_pred = P_ + process_var_;
        // Update.
        const double K = P_pred / (P_pred + observation_var_);
        x_ = x_ + K * (z - x_);
        P_ = (1.0 - K) * P_pred;
        return x_;
    }

private:
    const double process_var_;
    const double observation_var_;
    const double initial_state_;
    const double initial_variance_;
    double x_ = 0.0;
    double P_ = 0.0;
};

}  // namespace screamer

#endif
