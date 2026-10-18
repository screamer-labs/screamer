#ifndef SCREAMER_KALMAN_REGRESSION_H
#define SCREAMER_KALMAN_REGRESSION_H

// KalmanRegression: a regression whose intercept and slope follow random
// walks, estimated with a Kalman filter. The standard dynamic hedge ratio for
// pairs trading (Chan 2013, ch. 3).
//
//   observation:  y_t = alpha_t + beta_t * x_t + e_t,   e_t ~ N(0, obs_var)
//   state:        [alpha_t, beta_t] = [alpha_{t-1}, beta_{t-1}] + w_t,
//                 w_t ~ N(0, delta / (1 - delta) * I)
//
// Each step predicts y_t from the state before it, then updates the state:
//
//   R = P + Vw;  yhat = alpha + beta * x;  Q = h R h' + obs_var  (h = [1, x])
//   e = y - yhat;  K = R h' / Q;  state += K e;  P = (I - K h) R (I - K h)' + K obs_var K'
//
// Outputs, in order: beta and alpha after the update; the spread e, the
// prediction error made with the previous state (so it uses no information
// from step t in its hedge ratio); and sqrt(Q), the predicted standard
// deviation of that error, so e / sqrt(Q) is a ready z-score.
//
// The state starts at zero with variance prior_var on both coefficients. A
// unit variance is weak enough that the data sets the line within a few
// steps, and keeps the first updates well conditioned: with prices far from
// zero and little early movement the intercept and slope are nearly
// collinear, and a very large prior variance makes them numerically unstable.
// A zero prior (Chan's implementation) is stable but leaves a lasting bias
// when delta is small. The spread and its deviation are NaN on the first
// step, which has no prediction.
// delta sets how fast the coefficients may move. A large delta lets the
// intercept follow the spread itself, which removes the mean reversion the
// trade relies on; on log prices keep it small (1e-8 to 1e-5). obs_var is
// the variance of the spread, in squared units of y. A NaN in either input
// skips the step.
//
// The covariance update uses the Joseph form, which keeps P symmetric and
// positive semi-definite. O(1) per step.
//
// Robust update (huber, default inf = off): when |e| / sqrt(Q) > huber the
// update uses obs_var / w, w = huber / (|e| / sqrt(Q)), in place of obs_var in
// Q, the gain and the Joseph form (a Huber M-estimate step, Masreliez and
// Martin 1977), so an outlier's pull on the coefficients is bounded. The
// weight applies on every step with a prediction (not the first). A robust
// filter trusts its prediction, so it needs a prior that does not contradict
// the data, such as a wider prior_var. An infinite observation gets w = 0:
// the coefficients stay and P = R. The spread and
// spread_std outputs keep the raw innovation and the un-inflated sqrt(Q).

#include <cmath>
#include <limits>
#include <stdexcept>
#include "screamer/common/finite_settings.h"
#include "screamer/common/functor_base.h"
#include "screamer/common/float_info.h"

namespace screamer {

class KalmanRegression : public FunctorBase<KalmanRegression, 2, 4> {
public:
    KalmanRegression(double delta = 1e-6, double obs_var = 1e-3, double prior_var = 1.0,
                     double huber = std::numeric_limits<double>::infinity())
        : delta_(delta), obs_var_(obs_var), prior_var_(prior_var), huber_(huber),
          robust_(huber < std::numeric_limits<double>::infinity())
    {
        if (!(std::isfinite(prior_var_) && prior_var_ >= 0.0))
            throw std::invalid_argument("prior_var must be finite and non-negative.");
        if (!(delta_ > 0.0 && delta_ < 1.0))
            throw std::invalid_argument("delta must be in (0, 1).");
        if (!(std::isfinite(obs_var_) && obs_var_ > 0.0))
            throw std::invalid_argument("obs_var must be finite and positive.");
        require_finite("huber", huber_, kTakesPosInf);   // inf: off
        if (!(huber_ > 0.0))
            throw std::invalid_argument("huber must be > 0.");
        vw_ = delta_ / (1.0 - delta_);
        reset();
    }

    void reset() override {
        a_ = b_ = 0.0;
        p00_ = p11_ = prior_var_;
        p01_ = 0.0;
        n_ = 0;
    }

    ResultTuple call(const InputArray& in) override {
        const double y = in[0], x = in[1];
        const double nan = std::numeric_limits<double>::quiet_NaN();
        if (isnan2(y) || isnan2(x)) return {nan, nan, nan, nan};

        // Predict.
        const double r00 = p00_ + vw_, r01 = p01_, r11 = p11_ + vw_;
        const double rh0 = r00 + r01 * x;          // R h'
        const double rh1 = r01 + r11 * x;
        const double hrh = rh0 + rh1 * x;          // h R h'
        const double q = hrh + obs_var_;
        const double sq = std::sqrt(q);
        const double e = y - (a_ + b_ * x);
        double v = obs_var_, qv = q;               // observation variance used by the update
        if (robust_ && n_ > 0) {
            // Huber weight on the standardized innovation: inflate the
            // observation variance of an outlier to obs_var / w.
            const double az = std::fabs(e) / sq;
            if (az > huber_) {
                const double w = huber_ / az;
                if (!(w > 0.0)) {           // an infinite observation: no update
                    p00_ = r00; p01_ = r01; p11_ = r11;
                    ++n_;
                    return {b_, a_, e, sq};
                }
                v /= w;
                qv = hrh + v;
            }
        }
        // Update.
        const double k0 = rh0 / qv, k1 = rh1 / qv;
        a_ += k0 * e;
        b_ += k1 * e;
        // Joseph form: P = A R A' + K v K', A = I - K h.
        const double a00 = 1.0 - k0, a01 = -k0 * x, a10 = -k1, a11 = 1.0 - k1 * x;
        const double t00 = a00 * r00 + a01 * r01, t01 = a00 * r01 + a01 * r11;
        const double t10 = a10 * r00 + a11 * r01, t11 = a10 * r01 + a11 * r11;
        p00_ = t00 * a00 + t01 * a01 + k0 * k0 * v;
        p01_ = t00 * a10 + t01 * a11 + k0 * k1 * v;
        p11_ = t10 * a10 + t11 * a11 + k1 * k1 * v;
        ++n_;

        const bool predicted = n_ >= 2;
        return {b_, a_, predicted ? e : nan, predicted ? sq : nan};
    }

private:
    const double delta_, obs_var_, prior_var_, huber_;
    const bool robust_;              // huber finite
    double vw_ = 0.0;
    double a_ = 0.0, b_ = 0.0, p00_ = 0.0, p01_ = 0.0, p11_ = 0.0;
    long long n_ = 0;
};

}  // namespace screamer

#endif
