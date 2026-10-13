#ifndef SCREAMER_DETAIL_EHLERS_SINEWAVE_H
#define SCREAMER_DETAIL_EHLERS_SINEWAVE_H

// Building blocks from Ehlers, Rocket Science for Traders (2001), chapters
// "The Sinewave Indicator" and "The Instantaneous Trendline", as TA-Lib
// implements them in HT_SINE, HT_TRENDLINE and HT_TRENDMODE. Both read the
// smoothed price and SmoothPeriod of detail::HilbertCycle and are updated once
// per finite sample, during warm-up too, so their history is complete when
// the engine becomes ready.
//
// DcPhase (the dominant cycle phase), angles in degrees:
//   N = int(SmoothPeriod + .5)
//   Real = sum_{k<N} sin(360 k / N) Smooth[k]
//   Imag = sum_{k<N} cos(360 k / N) Smooth[k]
//   DCPhase = atan(Real / Imag) + 90 + 360 / SmoothPeriod   (the last term
//             compensates the one-sample lag of the smoothing WMA)
//   if Imag < 0: DCPhase += 180 ;  if DCPhase > 315: DCPhase -= 360
//   (if Imag == 0: DCPhase = DCPhase[1] + 90 sign(Real), then the same steps)
//   Sine = sin(DCPhase) ;  LeadSine = sin(DCPhase + 45)
// DCPhase increases with time, so LeadSine leads Sine by 1/8 cycle.
//
// Trendline:
//   ITrend    = mean of the last N raw prices
//   Trendline = (4 ITrend + 3 ITrend[1] + 2 ITrend[2] + ITrend[3]) / 10

#include <array>
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace screamer {
namespace detail {

// The longest DFT window: SmoothPeriod is clamped to [6, 50].
constexpr int kEhlersMaxCycle = 50;
// Past samples kept: the longest window less the newest sample, rounded up
// to a multiple of 4. Tap 0 (the newest sample) comes from the argument and
// taps 1 .. N-1 from the ring of earlier samples. The past taps are summed
// over N - 1 rounded up to a multiple of 4, with zero weights past the
// window, so the trip count changes only when N crosses a multiple of 4 and
// the loop branch stays predictable as N wanders.
constexpr int kEhlersSpan = 52;
// Table row: tap 0 plus kEhlersSpan past taps, padded.
constexpr int kEhlersRow = 56;

inline int ehlers_span(int n) { return (n + 2) & ~3; }  // (n - 1) rounded up to 4.

inline int ehlers_cycle_length(double smooth_period) {
    const double p = smooth_period + 0.5;
    if (!(p >= 0.0)) return 0;
    const int n = static_cast<int>(p);
    return n > kEhlersMaxCycle ? kEhlersMaxCycle : n;
}

// Per window length N in [0, 50], over taps k < kEhlersRow: sin and cos of
// 360 k / N interleaved (sincos[N][2k], sincos[N][2k + 1]), zero for k >= N.
// Built once.
struct EhlersTables {
    double sincos[kEhlersMaxCycle + 1][2 * kEhlersRow] = {};
    EhlersTables() {
        const double two_pi = std::atan(1.0) * 8.0;
        for (int n = 1; n <= kEhlersMaxCycle; ++n) {
            for (int k = 0; k < n; ++k) {
                const double a = static_cast<double>(k) * two_pi / static_cast<double>(n);
                sincos[n][2 * k] = std::sin(a);
                sincos[n][2 * k + 1] = std::cos(a);
            }
        }
    }
    static const EhlersTables& get() {
        static const EhlersTables t;
        return t;
    }
};

// The last kEhlersSpan samples, each stored twice (at pos and
// pos + kEhlersSpan) so that newest()[i] is the sample i steps before the
// newest one pushed, for every i < kEhlersSpan, with no wrap-around inside
// the loops.
class MirrorRing {
public:
    void reset() {
        buf_.fill(0.0);
        pos_ = 0;
    }
    void push(double v) {
        pos_ = (pos_ == 0) ? kEhlersSpan - 1 : pos_ - 1;
        buf_[pos_] = v;
        buf_[pos_ + kEhlersSpan] = v;
    }
    const double* newest() const { return buf_.data() + pos_; }

private:
    std::array<double, 2 * kEhlersSpan> buf_{};
    int pos_ = 0;
};

// atan(y / x) for x != 0, in radians, without branches, accurate to about
// 1 ulp: the Cephes rational approximation, with its three argument ranges
// chosen by selects and the ratio folded into one division. The DC phase's
// Real / Imag ratio jumps at random on noisy input, and the branches inside
// a library atan then mispredict and stall the Hilbert engine around them.
inline double atan_ratio(double y, double x) {
    const double ay = std::abs(y), ax = std::abs(x);
    const bool big = ay > 2.41421356237309504880 * ax;  // |y / x| > tan(3 pi / 8)
    const bool mid = !big && ay > 0.66 * ax;
    const double num = big ? -ax : (mid ? ay - ax : ay);
    const double den = big ? ay : (mid ? ay + ax : ax);
    const double y0 = big ? 1.57079632679489661923 : (mid ? 0.78539816339744830962 : 0.0);
    const double more = big ? 6.123233995736765886130e-17
                            : (mid ? 0.5 * 6.123233995736765886130e-17 : 0.0);
    const double r = num / den;
    const double z = r * r;
    const double p = (((-8.750608600031904122785e-1 * z - 1.615753718733365076637e1) * z
                       - 7.500855792314704667340e1) * z - 1.228866684490136173410e2) * z
                     - 6.485021904942025371773e1;
    const double q = ((((z + 2.485846490142306297962e1) * z + 1.650270098316988542046e2) * z
                       + 4.328810604912902668951e2) * z + 4.853903996359136964868e2) * z
                     + 1.945506571482613964425e2;
    const double t = y0 + ((r * (z * p / q) + r) + more);
    return ((y < 0.0) != (x < 0.0)) ? -t : t;
}

// The dominant cycle phase. update() runs the DFT; then exactly one of the
// two readers below is called per sample, the same one every time:
//   - advance_sine() for CycleSine: tracks the unit vector of DCPhase and
//     gives sine and leadsine without an arctangent;
//   - advance_degrees() for TrendMode: TA-Lib's DCPhase in degrees, and on
//     which side of leadsine the sine lies, without a sine or cosine.
// Both give TA-Lib's values up to rounding.
class DcPhase {
public:
    DcPhase() { reset(); }

    void reset() {
        smooth_.reset();
        re_ = im_ = 0.0;
        period_ = 0.0;
        cos_ = 1.0;  // DCPhase starts at 0 degrees, as in TA-Lib.
        sin_ = 0.0;
        phase_ = 0.0;
        prev_phase_ = 0.0;
        rate_ = 0.0;
        side_ = prev_side_ = 0;
    }

    void update(double smooth_price, double smooth_period) {
        period_ = smooth_period;
        const int n = ehlers_cycle_length(smooth_period);
        const double* sc = EhlersTables::get().sincos[n];
        const double* past = smooth_.newest();
        // s0..s3 sum the sine part and c0..c3 the cosine part over past taps
        // i = 0..3 (mod 4); four partial sums per part break the add chain.
        // Past tap i is window tap i + 1.
        double s0 = 0.0, s1 = 0.0, s2 = 0.0, s3 = 0.0;
        double c0 = 0.0, c1 = 0.0, c2 = 0.0, c3 = 0.0;
        const double* w = sc + 2;
        const double* const end = past + ehlers_span(n);
        for (const double* x = past; x != end; x += 4, w += 8) {
            s0 += w[0] * x[0]; c0 += w[1] * x[0];
            s1 += w[2] * x[1]; c1 += w[3] * x[1];
            s2 += w[4] * x[2]; c2 += w[5] * x[2];
            s3 += w[6] * x[3]; c3 += w[7] * x[3];
        }
        re_ = sc[0] * smooth_price + ((s0 + s1) + (s2 + s3));
        im_ = sc[1] * smooth_price + ((c0 + c1) + (c2 + c3));
        smooth_.push(smooth_price);
    }

    // CycleSine path. DCPhase = angle(Imag, Real) + 90 + 360 / SmoothPeriod,
    // the same angle as TA-Lib's atan(Real / Imag) plus 180 when Imag < 0,
    // built by rotating unit vectors. When Imag is 0, TA-Lib keeps the last
    // phase and turns it by 90 degrees toward the sign of Real.
    void advance_sine() {
        double c, s;
        if (std::abs(im_) > 0.0) {
            const double inv = 1.0 / std::sqrt(re_ * re_ + im_ * im_);
            c = im_ * inv;
            s = re_ * inv;
        } else if (re_ < 0.0) {
            c = sin_;
            s = -cos_;
        } else if (re_ > 0.0) {
            c = -sin_;
            s = cos_;
        } else {
            c = cos_;
            s = sin_;
        }
        // Turn by 90 degrees, then by 360 / SmoothPeriod.
        const double c90 = -s, s90 = c;
        if (period_ > 0.0) {
            const double d = 2.0 * M_PI / period_;
            const double cd = std::cos(d), sd = std::sin(d);
            cos_ = c90 * cd - s90 * sd;
            sin_ = s90 * cd + c90 * sd;
        } else {
            cos_ = c90;
            sin_ = s90;
        }
    }
    double sine() const { return sin_; }
    // sin(DCPhase + 45) = (sin(DCPhase) + cos(DCPhase)) / sqrt(2).
    double leadsine() const { return (sin_ + cos_) * 0.70710678118654752440; }

    // TrendMode path: TA-Lib's DCPhase in degrees, wrapped as TA-Lib wraps
    // it, and the side of the sine against the leadsine. sin(p) - sin(p + 45)
    // = -2 sin(22.5) cos(p + 22.5), so the sine lies above the leadsine for p
    // in (67.5, 247.5) (mod 360) and below it outside that range.
    void advance_degrees() {
        prev_phase_ = phase_;
        if (std::abs(im_) > 0.0) {
            phase_ = atan_ratio(re_, im_) * kRadToDeg;
        } else if (re_ < 0.0) {
            phase_ -= 90.0;
        } else if (re_ > 0.0) {
            phase_ += 90.0;
        }
        // Selects rather than branches: the sign of Imag flips at random on
        // noisy input, and a mispredicted branch would stall the engine.
        rate_ = (period_ > 0.0) ? 360.0 / period_ : 0.0;
        double ph = phase_ + 90.0 + rate_;
        ph += (im_ < 0.0) ? 180.0 : 0.0;
        phase_ = (ph > 315.0) ? ph - 360.0 : ph;
        // DCPhase stays in (-45, 360), so one turn maps it to [0, 360).
        const double p = (phase_ < 0.0) ? phase_ + 360.0 : phase_;
        prev_side_ = side_;
        side_ = (p > 67.5 && p < 247.5) ? 1 : ((p == 67.5 || p == 247.5) ? 0 : -1);
    }
    // 360 / SmoothPeriod in degrees per sample, the cycle's phase rate; 0
    // before the period is known. Set by advance_degrees().
    double rate() const { return rate_; }
    double phase() const { return phase_; }
    double prev_phase() const { return prev_phase_; }
    // +1 when sine > leadsine, -1 when sine < leadsine, 0 when equal. Both
    // start at 0, as TA-Lib's sine and leadsine start equal at 0.
    int side() const { return side_; }
    int prev_side() const { return prev_side_; }

private:
    MirrorRing smooth_;
    double re_ = 0.0, im_ = 0.0;
    double period_ = 0.0;
    double cos_ = 1.0, sin_ = 0.0;
    double phase_ = 0.0;
    double prev_phase_ = 0.0;
    double rate_ = 0.0;
    int side_ = 0, prev_side_ = 0;
    static constexpr double kRadToDeg = 57.295779513082320876798;
};

class CycleTrendline {
public:
    CycleTrendline() { reset(); }

    void reset() {
        price_.reset();
        count_ = 0;
        it1_ = it2_ = it3_ = 0.0;
        value_ = 0.0;
    }

    // One step per sample, after the engine update: the sum of the newest N
    // prices, newest first, as TA-Lib adds them.
    void update(double price, double smooth_period) {
        if (count_ < kEhlersMaxCycle) ++count_;
        price_.push(price);
        const int n = ehlers_cycle_length(smooth_period);
        const double* x = price_.newest();
        double sum = 0.0;
        for (int k = 0; k < n; ++k) sum += x[k];
        finish_sum(sum, n);
    }

    // The same in two steps, which InstantaneousTrendline uses: prepare()
    // needs only the price, so it is called before the engine update and its
    // running sums do not wait for the period; finish() picks the window.
    // Which form is faster depends on the work around it (measured on each
    // operator); both give identical values.
    void prepare(double price) {
        if (count_ < kEhlersMaxCycle) ++count_;
        // sum_[k] is the sum of the newest k prices, newest first, as TA-Lib
        // adds them.
        const double* past = price_.newest();
        sum_[0] = 0.0;
        sum_[1] = price;
        for (int k = 1; k < kEhlersMaxCycle; ++k) sum_[k + 1] = sum_[k] + past[k - 1];
        price_.push(price);
    }

    void finish(double smooth_period) {
        const int n = ehlers_cycle_length(smooth_period);
        finish_sum(sum_[n], n);
    }

    double value() const { return value_; }

private:
    void finish_sum(double sum, int n) {
        // The ring starts at zero, so before n samples have been seen the
        // sum holds only the samples seen and is divided by their count.
        const int d = n < count_ ? n : count_;
        double it = sum;
        if (d > 0) it /= static_cast<double>(d);
        value_ = (4.0 * it + 3.0 * it1_ + 2.0 * it2_ + it3_) / 10.0;
        it3_ = it2_;
        it2_ = it1_;
        it1_ = it;
    }

    MirrorRing price_;
    double sum_[kEhlersMaxCycle + 1] = {};
    int count_ = 0;
    double it1_ = 0.0, it2_ = 0.0, it3_ = 0.0;
    double value_ = 0.0;
};

}  // namespace detail
}  // namespace screamer

#endif  // SCREAMER_DETAIL_EHLERS_SINEWAVE_H
