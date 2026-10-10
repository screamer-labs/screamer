#ifndef SCREAMER_VALIDATION_SHARPE_H
#define SCREAMER_VALIDATION_SHARPE_H

#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>

namespace screamer { namespace validation {

    // Probabilistic and deflated Sharpe ratios (Bailey and Lopez de Prado,
    // 2012 and 2014). Both are probabilities that the true Sharpe ratio is
    // above a benchmark, given the observed per-step Sharpe ratio, the sample
    // length, and the skewness and kurtosis of the per-step pnl. The deflated
    // version raises the benchmark to the Sharpe ratio expected from the best
    // of N tried strategies, so it corrects for selection. These are analytic:
    // no resampling.

    // Standard normal CDF.
    inline double norm_cdf(double x) { return 0.5 * std::erfc(-x / std::sqrt(2.0)); }

    // Inverse standard normal CDF: Wichura's AS241 (PPND16), about 1e-16
    // relative accuracy, the same on every platform.
    inline double norm_ppf(double p) {
        if (!(p > 0.0 && p < 1.0)) {
            if (p == 0.0) return -std::numeric_limits<double>::infinity();
            if (p == 1.0) return std::numeric_limits<double>::infinity();
            return std::numeric_limits<double>::quiet_NaN();
        }
        const double q = p - 0.5;
        if (std::abs(q) <= 0.425) {
            const double r = 0.180625 - q * q;
            return q * (((((((2509.0809287301226727 * r + 33430.575583588128105) * r +
                    67265.770927008700853) * r + 45921.953931549871457) * r +
                    13731.693765509461125) * r + 1971.5909503065514427) * r +
                    133.14166789178437745) * r + 3.387132872796366608) /
                   (((((((5226.495278852545925 * r + 28729.085735721942674) * r +
                    39307.89580009271061) * r + 21213.794301586595867) * r +
                    5394.1960214247511077) * r + 687.1870074920579083) * r +
                    42.313330701600911252) * r + 1.0);
        }
        double r = q < 0.0 ? p : 1.0 - p;
        r = std::sqrt(-std::log(r));
        double val;
        if (r <= 5.0) {
            r -= 1.6;
            val = (((((((7.7454501427834140764e-4 * r + 0.0227238449892691845833) * r +
                    0.24178072517745061177) * r + 1.27045825245236838258) * r +
                    3.64784832476320460504) * r + 5.7694972214606914055) * r +
                    4.6303378461565452959) * r + 1.42343711074968357734) /
                  (((((((1.05075007164441684324e-9 * r + 5.475938084995344946e-4) * r +
                    0.0151986665636164571966) * r + 0.14810397642748007459) * r +
                    0.68976733498510000455) * r + 1.6763848301838038494) * r +
                    2.05319162663775882187) * r + 1.0);
        } else {
            r -= 5.0;
            val = (((((((2.01033439929228813265e-7 * r + 2.71155556874348757815e-5) * r +
                    0.0012426609473880784386) * r + 0.026532189526576123093) * r +
                    0.29656057182850489123) * r + 1.7848265399172913358) * r +
                    5.4637849111641143699) * r + 6.6579046435011037772) /
                  (((((((2.04426310338993978564e-15 * r + 1.4215117583164458887e-7) * r +
                    1.8463183175100546818e-5) * r + 7.868691311456132591e-4) * r +
                    0.0148753612908506148525) * r + 0.13692988092273580531) * r +
                    0.59983220655588793769) * r + 1.0);
        }
        return q < 0.0 ? -val : val;
    }

    // Per-step Sharpe ratio (mean over sample std), skewness, and kurtosis
    // (not excess; 3 for a normal) of the finite values of x. T is their count.
    struct SharpeStats { double sharpe, skew, kurt; std::size_t T; };

    inline SharpeStats sharpe_stats(const double* x, std::size_t n) {
        std::size_t T = 0;
        double mean = 0.0;
        for (std::size_t i = 0; i < n; ++i)
            if (std::isfinite(x[i])) { ++T; mean += x[i]; }
        if (T < 3) throw std::invalid_argument("need at least three finite pnl values.");
        mean /= static_cast<double>(T);
        double m2 = 0.0, m3 = 0.0, m4 = 0.0;
        for (std::size_t i = 0; i < n; ++i) {
            if (!std::isfinite(x[i])) continue;
            const double d = x[i] - mean, d2 = d * d;
            m2 += d2; m3 += d2 * d; m4 += d2 * d2;
        }
        const double Tn = static_cast<double>(T);
        if (!(m2 > 0.0)) throw std::invalid_argument("pnl has no dispersion.");
        const double sd = std::sqrt(m2 / (Tn - 1.0));
        const double pm2 = m2 / Tn;
        return {mean / sd, (m3 / Tn) / std::pow(pm2, 1.5), (m4 / Tn) / (pm2 * pm2), T};
    }

    // P(true Sharpe > benchmark) for an observed per-step Sharpe ratio.
    inline double probabilistic_sharpe(double sharpe, double benchmark, std::size_t T,
                                       double skew, double kurt) {
        const double var = 1.0 - skew * sharpe + (kurt - 1.0) / 4.0 * sharpe * sharpe;
        if (!(var > 0.0)) return std::numeric_limits<double>::quiet_NaN();
        return norm_cdf((sharpe - benchmark) * std::sqrt(static_cast<double>(T) - 1.0) /
                        std::sqrt(var));
    }

    // Expected maximum per-step Sharpe ratio of N strategies with no skill,
    // whose Sharpe ratios have the given variance across trials.
    inline double expected_max_sharpe(double trial_variance, std::size_t N) {
        if (N < 2 || !(trial_variance > 0.0)) return 0.0;
        const double gamma = 0.5772156649015329;     // Euler-Mascheroni
        const double n = static_cast<double>(N);
        return std::sqrt(trial_variance) *
               ((1.0 - gamma) * norm_ppf(1.0 - 1.0 / n) +
                gamma * norm_ppf(1.0 - 1.0 / (n * std::exp(1.0))));
    }


    // From a pnl series: P(true Sharpe > benchmark).
    inline double probabilistic_sharpe(const double* pnl, std::size_t n, double benchmark) {
        const SharpeStats st = sharpe_stats(pnl, n);
        return probabilistic_sharpe(st.sharpe, benchmark, st.T, st.skew, st.kurt);
    }

    // Deflated Sharpe ratio from a pnl series and the per-step Sharpe ratios of
    // every strategy tried (including this one). Returns the probability and,
    // through `benchmark`, the expected best Sharpe ratio of the trials.
    inline double deflated_sharpe(const double* pnl, std::size_t n,
                                  const double* trials, std::size_t n_trials,
                                  double& benchmark) {
        std::size_t N = 0;
        double mean = 0.0;
        for (std::size_t i = 0; i < n_trials; ++i)
            if (std::isfinite(trials[i])) { ++N; mean += trials[i]; }
        if (N < 2) throw std::invalid_argument(
            "deflated_sharpe needs the Sharpe ratios of at least two trials.");
        mean /= static_cast<double>(N);
        double var = 0.0;
        for (std::size_t i = 0; i < n_trials; ++i)
            if (std::isfinite(trials[i])) var += (trials[i] - mean) * (trials[i] - mean);
        var /= static_cast<double>(N - 1);
        benchmark = expected_max_sharpe(var, N);
        return probabilistic_sharpe(pnl, n, benchmark);
    }

}} // namespace screamer::validation

#endif // SCREAMER_VALIDATION_SHARPE_H
