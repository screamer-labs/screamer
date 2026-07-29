#ifndef SCREAMER_SIGNAL_EHLERS
#define SCREAMER_SIGNAL_EHLERS

#include <cmath>
#include <optional>
#include <stdexcept>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace screamer {

// Resolve the exactly-one-of {period, cutoff} argument pair to a period in
// samples. cutoff is a fraction of Nyquist in (0, 1); period = 2 / cutoff.
inline double ehlers_resolve_period(std::optional<double> period,
                                    std::optional<double> cutoff) {
    const int provided = (period.has_value() ? 1 : 0) + (cutoff.has_value() ? 1 : 0);
    if (provided != 1) {
        throw std::invalid_argument("Exactly one of period or cutoff must be provided.");
    }
    double p;
    if (period.has_value()) {
        p = period.value();
    } else {
        const double c = cutoff.value();
        if (!(c > 0.0 && c < 1.0)) {
            throw std::invalid_argument("cutoff must be in (0, 1).");
        }
        p = 2.0 / c;
    }
    if (!(p >= 2.0)) {
        throw std::invalid_argument("period must be at least 2 samples.");
    }
    return p;
}

// Ehlers 2-pole SuperSmoother lowpass transfer function.
inline void supersmoother_coeffs(double period, std::vector<double>& b,
                                 std::vector<double>& a) {
    const double a1 = std::exp(-std::sqrt(2.0) * M_PI / period);
    const double b1 = 2.0 * a1 * std::cos(std::sqrt(2.0) * M_PI / period);
    const double c2 = b1;
    const double c3 = -a1 * a1;
    const double c1 = 1.0 - c2 - c3;
    // IIRFilter's transposed-direct-form-II state update indexes b and a
    // up to the same n - 1 (see screamer::IIRFilter::process_scalar), so b
    // must be padded to match a's length even though the numerator is only
    // degree 1.
    b = {c1 / 2.0, c1 / 2.0, 0.0};
    a = {1.0, -c2, -c3};
}

// Ehlers 2-pole highpass transfer function (the roofing-filter highpass stage).
inline void ehlers_highpass2_coeffs(double period, std::vector<double>& b,
                                    std::vector<double>& a) {
    const double k = 0.707;
    const double w = k * 2.0 * M_PI / period;
    const double alpha = (std::cos(w) + std::sin(w) - 1.0) / std::cos(w);
    const double g = 1.0 - alpha / 2.0;
    const double om = 1.0 - alpha;
    b = {g * g, -2.0 * g * g, g * g};
    a = {1.0, -2.0 * om, om * om};
}

}  // namespace screamer

#endif  // SCREAMER_SIGNAL_EHLERS
