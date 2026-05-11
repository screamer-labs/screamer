#ifndef SCREAMER_BUTTER_MATH
#define SCREAMER_BUTTER_MATH

#include <iostream>
#include <vector>
#include <complex>
#include <cmath>
#include <stdexcept>
#include "screamer/signal/signal.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace screamer {

    ZPK ButterworthZPK(int N) {

        std::vector<std::complex<double>> zeros;

        std::vector<std::complex<double>> poles;
        for (int m = -N + 1; m < N; m += 2) {
            std::complex<double> pole = -std::exp(std::complex<double>(0, M_PI * m / (2.0 * N)));
            poles.push_back(pole);
        }

        return {zeros, poles, 1.0};
    }

    void butterworth_filter(int N, double cutoff_freq, std::vector<double>& b, std::vector<double>& a) {
        ZPK but = ButterworthZPK(N);

        // Pre-warp frequencies for digital filter design
        double fs = 2.0;
        double Wn = cutoff_freq;
        double warped = 2 * fs * std::tan(M_PI * Wn / fs);

        // lowpass
        ZPK but_lo = lp2lp_zpk(but.zeros, but.poles, but.gain, warped);

        // make digital
        ZPK digibut = bilinear_zpk(but_lo.zeros, but_lo.poles, but_lo.gain, 2.0);

        // convert (z,p,k) to (b,a)
        zpk2tf(digibut.zeros, digibut.poles, digibut.gain, b, a);
    }


    // High-pass Butterworth (b, a) coefficients via the standard
    // analog-prototype -> lp2hp -> bilinear -> tf pipeline.
    void butterworth_filter_hp(int N, double cutoff_freq,
                               std::vector<double>& b, std::vector<double>& a) {
        ZPK but = ButterworthZPK(N);
        double fs = 2.0;
        double warped = 2 * fs * std::tan(M_PI * cutoff_freq / fs);
        ZPK but_hp = lp2hp_zpk(but.zeros, but.poles, but.gain, warped);
        ZPK digi = bilinear_zpk(but_hp.zeros, but_hp.poles, but_hp.gain, 2.0);
        zpk2tf(digi.zeros, digi.poles, digi.gain, b, a);
    }


    // Band-pass Butterworth: produces a 2N-order filter.
    void butterworth_filter_bp(int N, double low_cutoff, double high_cutoff,
                               std::vector<double>& b, std::vector<double>& a) {
        if (!(low_cutoff > 0.0 && high_cutoff > low_cutoff && high_cutoff < 1.0)) {
            throw std::invalid_argument(
                "Bandpass cutoffs must satisfy 0 < low < high < 1 (Nyquist).");
        }
        ZPK but = ButterworthZPK(N);
        double fs = 2.0;
        double warped_lo = 2 * fs * std::tan(M_PI * low_cutoff / fs);
        double warped_hi = 2 * fs * std::tan(M_PI * high_cutoff / fs);
        double wo = std::sqrt(warped_lo * warped_hi);
        double bw = warped_hi - warped_lo;
        ZPK but_bp = lp2bp_zpk(but.zeros, but.poles, but.gain, wo, bw);
        ZPK digi = bilinear_zpk(but_bp.zeros, but_bp.poles, but_bp.gain, 2.0);
        zpk2tf(digi.zeros, digi.poles, digi.gain, b, a);
    }


    // Band-stop (notch) Butterworth: also produces a 2N-order filter.
    void butterworth_filter_bs(int N, double low_cutoff, double high_cutoff,
                               std::vector<double>& b, std::vector<double>& a) {
        if (!(low_cutoff > 0.0 && high_cutoff > low_cutoff && high_cutoff < 1.0)) {
            throw std::invalid_argument(
                "Bandstop cutoffs must satisfy 0 < low < high < 1 (Nyquist).");
        }
        ZPK but = ButterworthZPK(N);
        double fs = 2.0;
        double warped_lo = 2 * fs * std::tan(M_PI * low_cutoff / fs);
        double warped_hi = 2 * fs * std::tan(M_PI * high_cutoff / fs);
        double wo = std::sqrt(warped_lo * warped_hi);
        double bw = warped_hi - warped_lo;
        ZPK but_bs = lp2bs_zpk(but.zeros, but.poles, but.gain, wo, bw);
        ZPK digi = bilinear_zpk(but_bs.zeros, but_bs.poles, but_bs.gain, 2.0);
        zpk2tf(digi.zeros, digi.poles, digi.gain, b, a);
    }

} // namespace

#endif // include guards