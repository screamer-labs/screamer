#ifndef SCREAMER_DETAIL_HILBERT_CYCLE_H
#define SCREAMER_DETAIL_HILBERT_CYCLE_H

// HilbertCycle: John Ehlers' Homodyne Discriminator (Rocket Science for
// Traders, 2001). Estimates the dominant cycle period, the analytic-signal
// in-phase / quadrature components, the instantaneous phase, and the
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
//   phase     = atan2_deg(Q1, I1)   (instantaneous phase from I1/Q1)
//   amplitude = sqrt(I2*I2 + Q2*Q2)

#include <array>
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
        price_.fill(0.0);
        smooth_.fill(0.0);
        detrender_.fill(0.0);
        i1_.fill(0.0);
        q1_.fill(0.0);
        i2_ = prev_i2_ = 0.0;
        q2_ = prev_q2_ = 0.0;
        prev_re_ = 0.0;
        prev_im_ = 0.0;
        prev_period_ = 0.0;
        smooth_period_ = 0.0;
        phase_ = std::numeric_limits<double>::quiet_NaN();
        amplitude_ = std::numeric_limits<double>::quiet_NaN();
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
        if (prev_period_ > 0.0) {
            if (period > 1.5 * prev_period_) period = 1.5 * prev_period_;
            if (period < 0.67 * prev_period_) period = 0.67 * prev_period_;
        }
        if (period < 6.0) period = 6.0;
        if (period > 50.0) period = 50.0;
        period = 0.2 * period + 0.8 * prev_period_;
        smooth_period_ = 0.33 * period + 0.67 * smooth_period_;
        // Commit state.
        prev_i2_ = i2; prev_q2_ = q2;
        prev_re_ = re; prev_im_ = im;
        prev_period_ = period;
        i2_ = i2; q2_ = q2;
        phase_ = std::atan2(q1, i1) * 180.0 / M_PI;
        if (phase_ < 0.0) phase_ += 360.0;
        amplitude_ = std::sqrt(i2 * i2 + q2 * q2);
        n_++;
    }

    bool ready() const { return n_ > kWarmup; }
    double inphase() const { return (ready() && !last_input_nan_) ? i2_ : nan(); }
    double quadrature() const { return (ready() && !last_input_nan_) ? q2_ : nan(); }
    double period() const { return (ready() && !last_input_nan_) ? smooth_period_ : nan(); }
    double phase() const { return (ready() && !last_input_nan_) ? phase_ : nan(); }
    double amplitude() const { return (ready() && !last_input_nan_) ? amplitude_ : nan(); }

private:
    static constexpr int kWarmup = 40;  // discriminator settling; tuned via pure-tone test.
    static double nan() { return std::numeric_limits<double>::quiet_NaN(); }

    // 7-tap Hilbert weighting over a 7-sample history window [0..6].
    static double hilbert(const std::array<double, 7>& h) {
        return 0.0962 * h[0] + 0.5769 * h[2] - 0.5769 * h[4] - 0.0962 * h[6];
    }
    template <size_t N>
    static void shift(std::array<double, N>& a, double v) {
        for (size_t i = N - 1; i > 0; --i) a[i] = a[i - 1];
        a[0] = v;
    }

    std::array<double, 7> price_{};
    std::array<double, 7> smooth_{};
    std::array<double, 7> detrender_{};
    std::array<double, 7> i1_{};
    std::array<double, 7> q1_{};
    double i2_ = 0.0, prev_i2_ = 0.0;
    double q2_ = 0.0, prev_q2_ = 0.0;
    double prev_re_ = 0.0;
    double prev_im_ = 0.0;
    double prev_period_ = 0.0;
    double smooth_period_ = 0.0;
    double phase_ = nan();
    double amplitude_ = nan();
    int n_ = 0;
    bool last_input_nan_ = false;
};

}  // namespace detail
}  // namespace screamer

#endif  // SCREAMER_DETAIL_HILBERT_CYCLE_H
