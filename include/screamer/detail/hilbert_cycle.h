#ifndef SCREAMER_DETAIL_HILBERT_CYCLE_H
#define SCREAMER_DETAIL_HILBERT_CYCLE_H

// HilbertCycle: John Ehlers' Homodyne Discriminator (Rocket Science for
// Traders, 2001). Estimates the dominant cycle period, the in-phase and
// quadrature components I1 and Q1, the instantaneous phase, and the
// instantaneous amplitude of a sampled series, using only current and past
// samples. All accessors return NaN until the warm-up transient clears.
//
// Reference algorithm (EasyLanguage, angles in degrees):
//   Smooth    = (4*P[0] + 3*P[1] + 2*P[2] + P[3]) / 10
//   Detrender = (.0962*Smooth[0] + .5769*Smooth[2] - .5769*Smooth[4]
//                - .0962*Smooth[6]) * (.075*Period[1] + .54)
//   Q1 = (.0962*Detr[0] + .5769*Detr[2] - .5769*Detr[4] - .0962*Detr[6])
//        * (.075*Period[1] + .54)
//   I1 = Detrender[3]
//   jI = (.0962*I1[0] + .5769*I1[2] - .5769*I1[4] - .0962*I1[6])
//        * (.075*Period[1] + .54)
//   jQ = (.0962*Q1[0] + .5769*Q1[2] - .5769*Q1[4] - .0962*Q1[6])
//        * (.075*Period[1] + .54)
//   I2 = I1 - jQ ;  Q2 = Q1 + jI
//   I2 = .2*I2 + .8*I2[1] ;  Q2 = .2*Q2 + .8*Q2[1]
//   Re = I2*I2[1] + Q2*Q2[1] ;  Im = I2*Q2[1] - Q2*I2[1]
//   Re = .2*Re + .8*Re[1] ;  Im = .2*Im + .8*Im[1]
//   if Im!=0 and Re!=0: Period = 360 / atan_deg(Im/Re)
//   clamp Period to [0.67*Period[1], 1.5*Period[1]] then to [6, 50]
//   Period = .2*Period + .8*Period[1]
//   SmoothPeriod = .33*Period + .67*SmoothPeriod[1]
//
// Derived quantities (computed on demand, so an operator pays only for what
// it reads):
//   phase      = atan2_deg(Q1, I1), wrapped to [0, 360). It decreases by
//                about 360 / period per sample, as Ehlers' MAMA phase does.
//   inphase    = I1 / gI,  quadrature = Q1 / gQ,  amplitude = hypot of both.
//                (I1, Q1) is the pair TA-Lib HT_PHASOR returns. gI and gQ
//                are the steady-state gains of the smoothing WMA and of one
//                or two passes of the Hilbert FIR (with its .075*Period + .54
//                correction) at the current period, so a unit sine of a
//                period from about 7 to 50 reads I, Q and amplitude of
//                about 1. Ehlers takes |(I1, Q1)| as the cycle amplitude
//                (I1^2 + Q1^2 is the cycle power of his signal-to-noise
//                ratio) at the approximate unit gain of the .075*Period + .54
//                correction; dividing by the gains makes that gain exact.

#include <cstdint>
#include <cmath>
#include <cstddef>
#include <limits>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace screamer {
namespace detail {

class HilbertCycle {
public:
    HilbertCycle() { reset(); }

    void reset() {
        price_.reset();
        smooth_.reset();
        detrender_.reset();
        i1_.reset();
        q1_.reset();
        prev_i2_ = 0.0;
        prev_q2_ = 0.0;
        prev_re_ = 0.0;
        prev_im_ = 0.0;
        prev_period_ = 0.0;
        smooth_period_ = 0.0;
        adj_ = 0.54;
        n_ = 0;
        last_input_nan_ = false;
    }

    void update(double price) {
        if (std::isnan(price)) {
            // NaN policy ignore: leave all running state untouched; the
            // NaN itself is surfaced by the accessors via last_input_nan_.
            last_input_nan_ = true;
            return;
        }
        last_input_nan_ = false;
        shift(price_, price);
        const double sm = (4.0 * price_[0] + 3.0 * price_[1] + 2.0 * price_[2] + price_[3]) / 10.0;
        shift(smooth_, sm);
        const double adj = 0.075 * prev_period_ + 0.54;
        adj_ = adj;
        const double det = hilbert(smooth_) * adj;
        shift(detrender_, det);
        const double q1 = hilbert(detrender_) * adj;
        const double i1 = detrender_[3];
        shift(q1_, q1);
        shift(i1_, i1);
        const double jI = hilbert(i1_) * adj;
        const double jQ = hilbert(q1_) * adj;
        double i2 = i1 - jQ;
        double q2 = q1 + jI;
        i2 = 0.2 * i2 + 0.8 * prev_i2_;
        q2 = 0.2 * q2 + 0.8 * prev_q2_;
        double re = i2 * prev_i2_ + q2 * prev_q2_;
        double im = i2 * prev_q2_ - q2 * prev_i2_;
        re = 0.2 * re + 0.8 * prev_re_;
        im = 0.2 * im + 0.8 * prev_im_;
        double period = prev_period_;
        if (im != 0.0 && re != 0.0) {
            period = 360.0 / (std::atan(im / re) * 180.0 / M_PI);
        }
        // The clamps are written as selects, which give the same values as
        // the if-statements: on noisy input their outcome is unpredictable,
        // and a mispredicted branch here stalls the period feedback loop.
        if (prev_period_ > 0.0) {
            const double hi = 1.5 * prev_period_, lo = 0.67 * prev_period_;
            period = (period > hi) ? hi : period;
            period = (period < lo) ? lo : period;
        }
        period = (period < 6.0) ? 6.0 : period;
        period = (period > 50.0) ? 50.0 : period;
        period = 0.2 * period + 0.8 * prev_period_;
        smooth_period_ = 0.33 * period + 0.67 * smooth_period_;
        // Commit state.
        prev_i2_ = i2; prev_q2_ = q2;
        prev_re_ = re; prev_im_ = im;
        prev_period_ = period;
        n_++;
    }

    bool ready() const { return n_ > kWarmup; }
    // True when the last update() was a finite sample after warm-up.
    bool valid() const { return ready() && !last_input_nan_; }
    double period() const { return valid() ? smooth_period_ : nan(); }
    double phase() const {
        if (!valid()) return nan();
        double ph = std::atan2(q1_[0], i1_[0]) * 180.0 / M_PI;
        if (ph < 0.0) ph += 360.0;
        return ph;
    }
    // The scaled pair (inphase, quadrature); NaN until valid().
    void phasor(double& inphase, double& quadrature) const {
        if (!valid()) {
            inphase = quadrature = nan();
            return;
        }
        double gi, gq;
        gains(gi, gq);
        inphase = i1_[0] / gi;
        quadrature = q1_[0] / gq;
    }
    double amplitude() const {
        double i, q;
        phasor(i, q);
        return std::hypot(i, q);
    }

    // Raw state for the operators that build on the engine. These ignore the
    // warm-up and NaN gating: callers check valid() themselves.
    double raw_i1() const { return i1_[0]; }
    double raw_q1() const { return q1_[0]; }
    double smooth_price() const { return smooth_[0]; }
    double raw_smooth_period() const { return smooth_period_; }

private:
    static constexpr int kWarmup = 40;  // discriminator settling; tuned via pure-tone test.
    static double nan() { return std::numeric_limits<double>::quiet_NaN(); }

    // Steady-state gains of I1 and Q1 for a sine at the current period:
    //   W(w) = .4 + .3 e^{-iw} + .2 e^{-2iw} + .1 e^{-3iw}   (price smoothing)
    //   h(w) = 2 adj (.0962 sin 3w + .5769 sin w)             (|Hilbert FIR|)
    //   gI = |W| h,  gQ = |W| h^2
    // The period is held to the engine's [6, 50] range, where h > 0.
    void gains(double& gi, double& gq) const {
        double p = prev_period_;
        if (p < 6.0) p = 6.0;
        if (p > 50.0) p = 50.0;
        const double w = 2.0 * M_PI / p;
        const double c1 = std::cos(w), s1 = std::sin(w);
        const double c2 = 2.0 * c1 * c1 - 1.0, s2 = 2.0 * s1 * c1;
        const double c3 = c1 * (2.0 * c2 - 1.0), s3 = s1 * (2.0 * c2 + 1.0);
        const double wr = 0.4 + 0.3 * c1 + 0.2 * c2 + 0.1 * c3;
        const double wi = 0.3 * s1 + 0.2 * s2 + 0.1 * s3;
        const double h = 2.0 * adj_ * (0.0962 * s3 + 0.5769 * s1);
        gi = std::sqrt(wr * wr + wi * wi) * h;
        gq = gi * h;
    }

    // The last 8 values of a series, newest first: h[k] is the value k steps
    // back. Each value is stored twice (at p and p + 8) so h[k] needs no
    // wrap-around, and a push writes two slots instead of moving the array.
    struct History {
        double b[16] = {};
        int p = 0;
        void reset() {
            for (double& v : b) v = 0.0;
            p = 0;
        }
        void push(double v) {
            p = (p - 1) & 7;
            b[p] = v;
            b[p + 8] = v;
        }
        double operator[](int k) const { return b[p + k]; }
    };

    // 7-tap Hilbert weighting over the history window [0..6].
    static double hilbert(const History& h) {
        return 0.0962 * h[0] + 0.5769 * h[2] - 0.5769 * h[4] - 0.0962 * h[6];
    }
    static void shift(History& a, double v) { a.push(v); }

    History price_, smooth_, detrender_, i1_, q1_;
    double prev_i2_ = 0.0;
    double prev_q2_ = 0.0;
    double prev_re_ = 0.0;
    double prev_im_ = 0.0;
    double prev_period_ = 0.0;
    double smooth_period_ = 0.0;
    double adj_ = 0.54;
    std::int64_t n_ = 0;  // 64-bit: a live stream passes 2**31 events
    bool last_input_nan_ = false;
};

}  // namespace detail
}  // namespace screamer

#endif  // SCREAMER_DETAIL_HILBERT_CYCLE_H
