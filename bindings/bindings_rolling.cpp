#include <pybind11/pybind11.h>
#include <pybind11/stl.h> // Required for std::optional support
#include "screamer/common/base.h"
#include "screamer/common/eval_op.h"
#include "screamer/rolling_sum.h"
#include "screamer/rolling_mean.h"
#include "screamer/rolling_var.h"
#include "screamer/rolling_std.h"
#include "screamer/rolling_skew.h"
#include "screamer/rolling_kurt.h"
#include "screamer/rolling_zscore.h"
#include "screamer/rolling_min.h"
#include "screamer/rolling_max.h"
#include "screamer/rolling_median.h"
#include "screamer/rolling_median_ad.h"
#include "screamer/hampel.h"
#include "screamer/impulse_clip.h"
#include "screamer/rolling_quantile.h"
#include "screamer/rolling_rms.h"
#include "screamer/rolling_poly1.h"
#include "screamer/rolling_poly2.h"
#include "screamer/rolling_sigma_clip.h"
#include "screamer/rolling_ou.h"
#include "screamer/rolling_rsi.h"
#include "screamer/rolling_min_max.h"
#include "screamer/bollinger_bands.h"
#include "screamer/rolling_argmin.h"
#include "screamer/rolling_argmax.h"
#include "screamer/rolling_range.h"
#include "screamer/rolling_mad.h"
#include "screamer/rolling_iqr.h"
#include "screamer/wma.h"
#include "screamer/dema.h"
#include "screamer/tema.h"
#include "screamer/trima.h"
#include "screamer/hull_ma.h"
#include "screamer/kama.h"
#include "screamer/macd.h"
#include "screamer/williams_r.h"
#include "screamer/stoch.h"
#include "screamer/trix.h"
#include "screamer/bop.h"
#include "screamer/cci.h"
#include "screamer/ultimate_oscillator.h"
#include "screamer/stoch_rsi.h"
#include "screamer/parkinson.h"
#include "screamer/garman_klass.h"
#include "screamer/rogers_satchell.h"
#include "screamer/true_range.h"
#include "screamer/atr.h"
#include "screamer/natr.h"
#include "screamer/donchian_channels.h"
#include "screamer/keltner_channels.h"
#include "screamer/yang_zhang.h"
#include "screamer/adx.h"
#include "screamer/vwap.h"
#include "screamer/obv.h"
#include "screamer/ad.h"
#include "screamer/adosc.h"
#include "screamer/mfi.h"
#include "screamer/rolling_tsf.h"
#include "screamer/rolling_rank.h"
#include "screamer/rolling_percentile.h"
#include "screamer/rolling_hurst.h"

namespace py = pybind11;

void init_bindings_rolling(py::module& m) {

    py::class_<screamer::RollingMean, screamer::ScreamerBase>(m, "RollingMean")
        .def(py::init<int, const std::string&>(), 
            py::arg("window_size") = 20, 
            py::arg("start_policy") = "strict")
        .def("__call__", &screamer::RollingMean::operator(), py::arg("value"))
        .def("reset", &screamer::RollingMean::reset, "Reset to the initial state.");

    py::class_<screamer::RollingRms, screamer::ScreamerBase>(m, "RollingRms")
        .def(py::init<int, const std::string&>(),
            py::arg("window_size") = 20,
            py::arg("start_policy") = "strict")
        .def("__call__", &screamer::RollingRms::operator(), py::arg("value"))
        .def("reset", &screamer::RollingRms::reset, "Reset to the initial state.");

    py::class_<screamer::RollingSum, screamer::ScreamerBase>(m, "RollingSum")
        .def(py::init<int, const std::string&>(), 
            py::arg("window_size") = 20, 
            py::arg("start_policy") = "strict")
        .def("__call__", &screamer::RollingSum::operator(), py::arg("value"))
        .def("reset", &screamer::RollingSum::reset, "Reset to the initial state.");

    py::class_<screamer::RollingStd, screamer::ScreamerBase>(m, "RollingStd")
        .def(py::init<int, const std::string&>(),
            py::arg("window_size") = 20,
            py::arg("start_policy") = "strict")
        .def("__call__", &screamer::RollingStd::operator(), py::arg("value"))
        .def("reset", &screamer::RollingStd::reset, "Reset to the initial state.");

    py::class_<screamer::RollingVar, screamer::ScreamerBase>(m, "RollingVar")
        .def(py::init<int, const std::string&>(),
            py::arg("window_size") = 20,
            py::arg("start_policy") = "strict")
        .def("__call__", &screamer::RollingVar::operator(), py::arg("value"))
        .def("reset", &screamer::RollingVar::reset, "Reset to the initial state.");

    py::class_<screamer::RollingSkew, screamer::ScreamerBase>(m, "RollingSkew")
        .def(py::init<int, const std::string&>(),
            py::arg("window_size") = 20,
            py::arg("start_policy") = "strict")
        .def("__call__", &screamer::RollingSkew::operator(), py::arg("value"))
        .def("reset", &screamer::RollingSkew::reset, "Reset to the initial state.");

    py::class_<screamer::RollingKurt, screamer::ScreamerBase>(m, "RollingKurt")
        .def(py::init<int, const std::string&>(),
            py::arg("window_size") = 20,
            py::arg("start_policy") = "strict")
        .def("__call__", &screamer::RollingKurt::operator(), py::arg("value"))
        .def("reset", &screamer::RollingKurt::reset, "Reset to the initial state.");

    py::class_<screamer::RollingMin, screamer::ScreamerBase>(m, "RollingMin")
        .def(py::init<int>(), py::arg("window_size") = 20)
        .def("__call__", &screamer::RollingMin::operator(), py::arg("value"))
        .def("reset", &screamer::RollingMin::reset, "Reset to the initial state.");

    py::class_<screamer::RollingMax, screamer::ScreamerBase>(m, "RollingMax")
        .def(py::init<int>(), py::arg("window_size") = 20)
        .def("__call__", &screamer::RollingMax::operator(), py::arg("value"))
        .def("reset", &screamer::RollingMax::reset, "Reset to the initial state.");

    // Position of the rolling minimum / maximum within the window.
    // 0 = oldest sample, window_size-1 = newest. Same monotonic-deque
    // primitive as RollingMin / RollingMax, exposed via the front
    // element's window offset.
    py::class_<screamer::RollingArgmin, screamer::ScreamerBase>(m, "RollingArgmin")
        .def(py::init<int>(), py::arg("window_size") = 20)
        .def("__call__", &screamer::RollingArgmin::operator(), py::arg("value"))
        .def("reset", &screamer::RollingArgmin::reset, "Reset to the initial state.");

    py::class_<screamer::RollingArgmax, screamer::ScreamerBase>(m, "RollingArgmax")
        .def(py::init<int>(), py::arg("window_size") = 20)
        .def("__call__", &screamer::RollingArgmax::operator(), py::arg("value"))
        .def("reset", &screamer::RollingArgmax::reset, "Reset to the initial state.");

    // RollingRange: max - min. Two monotonic deques internally,
    // composed at the primitive level (same algorithm RollingMinMax
    // runs, returned as a single scalar instead of a tuple).
    py::class_<screamer::RollingRange, screamer::ScreamerBase>(m, "RollingRange")
        .def(py::init<int>(), py::arg("window_size") = 20)
        .def("__call__", &screamer::RollingRange::operator(), py::arg("value"))
        .def("reset", &screamer::RollingRange::reset, "Reset to the initial state.");

    // Mean absolute deviation, mean(|x - rolling_mean|). O(W) per step
    // (provably no closed-form O(1) exists; the moving mean
    // re-evaluates all W abs-deviations each step).
    py::class_<screamer::RollingMad, screamer::ScreamerBase>(m, "RollingMad")
        .def(py::init<int, const std::string&>(),
            py::arg("window_size") = 20,
            py::arg("start_policy") = "strict")
        .def("__call__", &screamer::RollingMad::operator(), py::arg("value"))
        .def("reset", &screamer::RollingMad::reset, "Reset to the initial state.");

    // Robust scale: the median absolute deviation, median(|x - median|), over the
    // trailing window. Unlike RollingMad (mean absolute deviation) it is robust to
    // outliers, and it is the scale primitive behind Hampel and ImpulseClip.
    py::class_<screamer::RollingMedianAD, screamer::ScreamerBase>(m, "RollingMedianAD")
        .def(py::init<int, const std::string&>(),
            py::arg("window_size") = 20,
            py::arg("start_policy") = "strict")
        .def("__call__", &screamer::RollingMedianAD::operator(), py::arg("value"))
        .def("reset", &screamer::RollingMedianAD::reset, "Reset to the initial state.");

    // Inter-quartile range = q75 - q25. Single shared OST queried
    // twice (vs. two RollingQuantile instances which would use two
    // independent trees). Same O(log W) per step, half the memory
    // and inserts.
    py::class_<screamer::RollingIqr, screamer::ScreamerBase>(m, "RollingIqr")
        .def(py::init<int>(), py::arg("window_size") = 20)
        .def("__call__", &screamer::RollingIqr::operator(), py::arg("value"))
        .def("reset", &screamer::RollingIqr::reset, "Reset to the initial state.");

    // WMA: linearly-weighted moving average. O(1) per step via the
    // identity W[t] - W[t-1] = w*x[t] - S[t-1] where S is the simple
    // rolling sum of the previous window.
    py::class_<screamer::WMA, screamer::ScreamerBase>(m, "WMA")
        .def(py::init<int, const std::string&>(),
            py::arg("window_size") = 20,
            py::arg("start_policy") = "strict")
        .def("__call__", &screamer::WMA::operator(), py::arg("value"))
        .def("reset", &screamer::WMA::reset, "Reset to the initial state.");

    // DEMA / TEMA: double / triple exponential MA (Mulloy 1994). Pure
    // composition of 2 / 3 chained EwMean instances.
    py::class_<screamer::DEMA, screamer::ScreamerBase>(m, "DEMA")
        .def(
          py::init<
               std::optional<double>,
               std::optional<double>,
               std::optional<double>,
               std::optional<double>
          >(),
          py::arg("com") = std::nullopt,
          py::arg("span") = std::nullopt,
          py::arg("halflife") = std::nullopt,
          py::arg("alpha") = std::nullopt
        )
        .def("__call__", &screamer::DEMA::operator(), py::arg("value"))
        .def("reset", &screamer::DEMA::reset, "Reset to the initial state.");

    py::class_<screamer::TEMA, screamer::ScreamerBase>(m, "TEMA")
        .def(
          py::init<
               std::optional<double>,
               std::optional<double>,
               std::optional<double>,
               std::optional<double>
          >(),
          py::arg("com") = std::nullopt,
          py::arg("span") = std::nullopt,
          py::arg("halflife") = std::nullopt,
          py::arg("alpha") = std::nullopt
        )
        .def("__call__", &screamer::TEMA::operator(), py::arg("value"))
        .def("reset", &screamer::TEMA::reset, "Reset to the initial state.");

    // TRIMA: triangular MA, SMA(SMA(x)). Pure composition of two
    // detail::RollingMean instances. Strict warmup enforced by counter.
    py::class_<screamer::TRIMA, screamer::ScreamerBase>(m, "TRIMA")
        .def(py::init<int>(), py::arg("window_size") = 20)
        .def("__call__", &screamer::TRIMA::operator(), py::arg("value"))
        .def("reset", &screamer::TRIMA::reset, "Reset to the initial state.");

    // HullMA: WMA(2*WMA(x, n/2) - WMA(x, n), sqrt(n)). Pure composition
    // of three WMA instances. Inner WMAs use "expanding" so they don't
    // emit NaN; HullMA enforces strict warmup itself.
    py::class_<screamer::HullMA, screamer::ScreamerBase>(m, "HullMA")
        .def(py::init<int>(), py::arg("window_size") = 20)
        .def("__call__", &screamer::HullMA::operator(), py::arg("value"))
        .def("reset", &screamer::HullMA::reset, "Reset to the initial state.");

    // KAMA: Kaufman's Adaptive MA. Smoothing constant adapts to the
    // efficiency ratio (net displacement / total absolute travel).
    py::class_<screamer::KAMA, screamer::ScreamerBase>(m, "KAMA")
        .def(py::init<int, int, int>(),
            py::arg("window_size") = 10,
            py::arg("fast") = 2,
            py::arg("slow") = 30)
        .def("__call__", &screamer::KAMA::operator(), py::arg("value"))
        .def("reset", &screamer::KAMA::reset, "Reset to the initial state.");

    // MACD: (macd, signal, histogram). 1->3 functor composing three
    // pandas adjust=True EMAs (our EwMean).
    py::class_<screamer::MACD, screamer::EvalOp>(m, "MACD")
        .def(py::init<int, int, int>(),
            py::arg("fast") = 12,
            py::arg("slow") = 26,
            py::arg("signal") = 9)
        .def("__call__", &screamer::MACD::handle_input)
        .def("reset", &screamer::MACD::reset, "Reset to the initial state.");

    // WilliamsR: 3->1, takes (high, low, close), returns %R in [-100, 0].
    py::class_<screamer::WilliamsR, screamer::EvalOp>(m, "WilliamsR")
        .def(py::init<int>(), py::arg("window_size") = 14)
        .def("__call__", &screamer::WilliamsR::handle_input)
        .def("reset", &screamer::WilliamsR::reset, "Reset to the initial state.");

    // Stoch: 3->2, takes (high, low, close), returns (%K, %D). With
    // smooth_k=1 this is the "fast" stochastic (Lane's original);
    // with smooth_k>=2 it is the "slow" stochastic (talib.STOCH).
    py::class_<screamer::Stoch, screamer::EvalOp>(m, "Stoch")
        .def(py::init<int, int, int>(),
            py::arg("fastk_period") = 14,
            py::arg("smooth_k") = 3,
            py::arg("d") = 3)
        .def("__call__", &screamer::Stoch::handle_input)
        .def("reset", &screamer::Stoch::reset, "Reset to the initial state.");

    // TRIX: 100 * 1-period ROC of triple-smoothed EMA. Composes
    // three EwMean instances and tracks the previous ema3 for the
    // final ratio.
    py::class_<screamer::TRIX, screamer::ScreamerBase>(m, "TRIX")
        .def(py::init<int>(), py::arg("span") = 14)
        .def("__call__", &screamer::TRIX::operator(), py::arg("value"))
        .def("reset", &screamer::TRIX::reset, "Reset to the initial state.");

    // BOP: Balance of Power. 4 -> 1 on (open, high, low, close).
    py::class_<screamer::BOP, screamer::EvalOp>(m, "BOP")
        .def(py::init<>())
        .def("__call__", &screamer::BOP::handle_input)
        .def("reset", &screamer::BOP::reset, "Reset to the initial state.");

    // CCI: Commodity Channel Index. 3 -> 1 on (high, low, close).
    py::class_<screamer::CCI, screamer::EvalOp>(m, "CCI")
        .def(py::init<int>(), py::arg("window_size") = 14)
        .def("__call__", &screamer::CCI::handle_input)
        .def("reset", &screamer::CCI::reset, "Reset to the initial state.");

    // UltimateOscillator: 3 -> 1 on (high, low, close); weighted
    // average over three timeframes.
    py::class_<screamer::UltimateOscillator, screamer::EvalOp>(m, "UltimateOscillator")
        .def(py::init<int, int, int>(),
            py::arg("period1") = 7,
            py::arg("period2") = 14,
            py::arg("period3") = 28)
        .def("__call__", &screamer::UltimateOscillator::handle_input)
        .def("reset", &screamer::UltimateOscillator::reset,
             "Reset to the initial state.");

    // StochRSI: 1 -> 2; Stochastic of RSI. Default smooth_k=1 (fast,
    // matching TA-Lib's STOCHRSI); set smooth_k >= 2 for slow form.
    py::class_<screamer::StochRSI, screamer::EvalOp>(m, "StochRSI")
        .def(py::init<int, int, int, int>(),
            py::arg("rsi_period") = 14,
            py::arg("stoch_period") = 14,
            py::arg("smooth_k") = 1,
            py::arg("d") = 3)
        .def("__call__", &screamer::StochRSI::handle_input)
        .def("reset", &screamer::StochRSI::reset, "Reset to the initial state.");

    // ----- Range-based volatility estimators -----
    // Each estimator has a *Var (variance) form and a *Vol (= sqrt of
    // *Var) form, in both Rolling and EW smoothing variants.

    // Parkinson (1980): H, L. 2 -> 1.
    py::class_<screamer::RollingParkinsonVar, screamer::EvalOp>(m, "RollingParkinsonVar")
        .def(py::init<int>(), py::arg("window_size") = 20)
        .def("__call__", &screamer::RollingParkinsonVar::handle_input)
        .def("reset", &screamer::RollingParkinsonVar::reset, "Reset.");

    py::class_<screamer::RollingParkinsonVol, screamer::EvalOp>(m, "RollingParkinsonVol")
        .def(py::init<int>(), py::arg("window_size") = 20)
        .def("__call__", &screamer::RollingParkinsonVol::handle_input)
        .def("reset", &screamer::RollingParkinsonVol::reset, "Reset.");

    py::class_<screamer::EwParkinsonVar, screamer::EvalOp>(m, "EwParkinsonVar")
        .def(py::init<std::optional<double>, std::optional<double>,
                       std::optional<double>, std::optional<double>>(),
             py::arg("com") = std::nullopt, py::arg("span") = std::nullopt,
             py::arg("halflife") = std::nullopt, py::arg("alpha") = std::nullopt)
        .def("__call__", &screamer::EwParkinsonVar::handle_input)
        .def("reset", &screamer::EwParkinsonVar::reset, "Reset.");

    py::class_<screamer::EwParkinsonVol, screamer::EvalOp>(m, "EwParkinsonVol")
        .def(py::init<std::optional<double>, std::optional<double>,
                       std::optional<double>, std::optional<double>>(),
             py::arg("com") = std::nullopt, py::arg("span") = std::nullopt,
             py::arg("halflife") = std::nullopt, py::arg("alpha") = std::nullopt)
        .def("__call__", &screamer::EwParkinsonVol::handle_input)
        .def("reset", &screamer::EwParkinsonVol::reset, "Reset.");

    // Garman-Klass (1980): O, H, L, C. 4 -> 1.
    py::class_<screamer::RollingGarmanKlassVar, screamer::EvalOp>(m, "RollingGarmanKlassVar")
        .def(py::init<int>(), py::arg("window_size") = 20)
        .def("__call__", &screamer::RollingGarmanKlassVar::handle_input)
        .def("reset", &screamer::RollingGarmanKlassVar::reset, "Reset.");

    py::class_<screamer::RollingGarmanKlassVol, screamer::EvalOp>(m, "RollingGarmanKlassVol")
        .def(py::init<int>(), py::arg("window_size") = 20)
        .def("__call__", &screamer::RollingGarmanKlassVol::handle_input)
        .def("reset", &screamer::RollingGarmanKlassVol::reset, "Reset.");

    py::class_<screamer::EwGarmanKlassVar, screamer::EvalOp>(m, "EwGarmanKlassVar")
        .def(py::init<std::optional<double>, std::optional<double>,
                       std::optional<double>, std::optional<double>>(),
             py::arg("com") = std::nullopt, py::arg("span") = std::nullopt,
             py::arg("halflife") = std::nullopt, py::arg("alpha") = std::nullopt)
        .def("__call__", &screamer::EwGarmanKlassVar::handle_input)
        .def("reset", &screamer::EwGarmanKlassVar::reset, "Reset.");

    py::class_<screamer::EwGarmanKlassVol, screamer::EvalOp>(m, "EwGarmanKlassVol")
        .def(py::init<std::optional<double>, std::optional<double>,
                       std::optional<double>, std::optional<double>>(),
             py::arg("com") = std::nullopt, py::arg("span") = std::nullopt,
             py::arg("halflife") = std::nullopt, py::arg("alpha") = std::nullopt)
        .def("__call__", &screamer::EwGarmanKlassVol::handle_input)
        .def("reset", &screamer::EwGarmanKlassVol::reset, "Reset.");

    // Rogers-Satchell (1991): O, H, L, C; drift-robust. 4 -> 1.
    py::class_<screamer::RollingRogersSatchellVar, screamer::EvalOp>(m, "RollingRogersSatchellVar")
        .def(py::init<int>(), py::arg("window_size") = 20)
        .def("__call__", &screamer::RollingRogersSatchellVar::handle_input)
        .def("reset", &screamer::RollingRogersSatchellVar::reset, "Reset.");

    py::class_<screamer::RollingRogersSatchellVol, screamer::EvalOp>(m, "RollingRogersSatchellVol")
        .def(py::init<int>(), py::arg("window_size") = 20)
        .def("__call__", &screamer::RollingRogersSatchellVol::handle_input)
        .def("reset", &screamer::RollingRogersSatchellVol::reset, "Reset.");

    py::class_<screamer::EwRogersSatchellVar, screamer::EvalOp>(m, "EwRogersSatchellVar")
        .def(py::init<std::optional<double>, std::optional<double>,
                       std::optional<double>, std::optional<double>>(),
             py::arg("com") = std::nullopt, py::arg("span") = std::nullopt,
             py::arg("halflife") = std::nullopt, py::arg("alpha") = std::nullopt)
        .def("__call__", &screamer::EwRogersSatchellVar::handle_input)
        .def("reset", &screamer::EwRogersSatchellVar::reset, "Reset.");

    py::class_<screamer::EwRogersSatchellVol, screamer::EvalOp>(m, "EwRogersSatchellVol")
        .def(py::init<std::optional<double>, std::optional<double>,
                       std::optional<double>, std::optional<double>>(),
             py::arg("com") = std::nullopt, py::arg("span") = std::nullopt,
             py::arg("halflife") = std::nullopt, py::arg("alpha") = std::nullopt)
        .def("__call__", &screamer::EwRogersSatchellVol::handle_input)
        .def("reset", &screamer::EwRogersSatchellVol::reset, "Reset.");

    // TrueRange / ATR / NATR (Wilder family). 3 -> 1 on (high, low, close).
    py::class_<screamer::TrueRange, screamer::EvalOp>(m, "TrueRange")
        .def(py::init<>())
        .def("__call__", &screamer::TrueRange::handle_input)
        .def("reset", &screamer::TrueRange::reset, "Reset to the initial state.");

    py::class_<screamer::ATR, screamer::EvalOp>(m, "ATR")
        .def(py::init<int>(), py::arg("window_size") = 14)
        .def("__call__", &screamer::ATR::handle_input)
        .def("reset", &screamer::ATR::reset, "Reset to the initial state.");

    py::class_<screamer::NATR, screamer::EvalOp>(m, "NATR")
        .def(py::init<int>(), py::arg("window_size") = 14)
        .def("__call__", &screamer::NATR::handle_input)
        .def("reset", &screamer::NATR::reset, "Reset to the initial state.");

    // Donchian / Keltner channels (envelope-style indicators).
    py::class_<screamer::DonchianChannels, screamer::EvalOp>(m, "DonchianChannels")
        .def(py::init<int>(), py::arg("window_size") = 20)
        .def("__call__", &screamer::DonchianChannels::handle_input)
        .def("reset", &screamer::DonchianChannels::reset, "Reset.");

    py::class_<screamer::KeltnerChannels, screamer::EvalOp>(m, "KeltnerChannels")
        .def(py::init<int, double>(),
             py::arg("window_size") = 20, py::arg("num_atr") = 2.0)
        .def("__call__", &screamer::KeltnerChannels::handle_input)
        .def("reset", &screamer::KeltnerChannels::reset, "Reset.");

    // Yang-Zhang volatility (the most efficient classical range-based
    // estimator; handles both drift and overnight gaps).
    py::class_<screamer::RollingYangZhangVar, screamer::EvalOp>(m, "RollingYangZhangVar")
        .def(py::init<int>(), py::arg("window_size") = 20)
        .def("__call__", &screamer::RollingYangZhangVar::handle_input)
        .def("reset", &screamer::RollingYangZhangVar::reset, "Reset.");

    py::class_<screamer::RollingYangZhangVol, screamer::EvalOp>(m, "RollingYangZhangVol")
        .def(py::init<int>(), py::arg("window_size") = 20)
        .def("__call__", &screamer::RollingYangZhangVol::handle_input)
        .def("reset", &screamer::RollingYangZhangVol::reset, "Reset.");

    // ADX (Wilder, 1978). 3 -> 3 on (high, low, close) returning
    // (plus_di, minus_di, adx). Match talib.PLUS_DI / MINUS_DI / ADX.
    py::class_<screamer::ADX, screamer::EvalOp>(m, "ADX")
        .def(py::init<int>(), py::arg("window_size") = 14)
        .def("__call__", &screamer::ADX::handle_input)
        .def("reset", &screamer::ADX::reset, "Reset to the initial state.");

    // Volume-aware indicators.
    py::class_<screamer::RollingVWAP, screamer::EvalOp>(m, "RollingVWAP")
        .def(py::init<int>(), py::arg("window_size") = 20)
        .def("__call__", &screamer::RollingVWAP::handle_input)
        .def("reset", &screamer::RollingVWAP::reset, "Reset.");

    py::class_<screamer::OBV, screamer::EvalOp>(m, "OBV")
        .def(py::init<>())
        .def("__call__", &screamer::OBV::handle_input)
        .def("reset", &screamer::OBV::reset, "Reset.");

    py::class_<screamer::AD, screamer::EvalOp>(m, "AD")
        .def(py::init<>())
        .def("__call__", &screamer::AD::handle_input)
        .def("reset", &screamer::AD::reset, "Reset.");

    py::class_<screamer::ADOSC, screamer::EvalOp>(m, "ADOSC")
        .def(py::init<int, int>(), py::arg("fast") = 3, py::arg("slow") = 10)
        .def("__call__", &screamer::ADOSC::handle_input)
        .def("reset", &screamer::ADOSC::reset, "Reset.");

    py::class_<screamer::MFI, screamer::EvalOp>(m, "MFI")
        .def(py::init<int>(), py::arg("window_size") = 14)
        .def("__call__", &screamer::MFI::handle_input)
        .def("reset", &screamer::MFI::reset, "Reset.");

    // Time-Series Forecast (TA-Lib's TSF): linear regression vs
    // time projected one step ahead. Composes detail::RollingSum.
    py::class_<screamer::RollingTSF, screamer::ScreamerBase>(m, "RollingTSF")
        .def(py::init<int>(), py::arg("window_size") = 20)
        .def("__call__", &screamer::RollingTSF::operator(), py::arg("value"))
        .def("reset", &screamer::RollingTSF::reset, "Reset.");

    // RollingRank / RollingPercentile: position of latest value in
    // the trailing window. pandas-style "average" tie rule.
    py::class_<screamer::RollingRank, screamer::ScreamerBase>(m, "RollingRank")
        .def(py::init<int>(), py::arg("window_size") = 20)
        .def("__call__", &screamer::RollingRank::operator(), py::arg("value"))
        .def("reset", &screamer::RollingRank::reset, "Reset.");

    py::class_<screamer::RollingPercentile, screamer::ScreamerBase>(m, "RollingPercentile")
        .def(py::init<int>(), py::arg("window_size") = 20)
        .def("__call__", &screamer::RollingPercentile::operator(), py::arg("value"))
        .def("reset", &screamer::RollingPercentile::reset, "Reset.");

    // RollingHurst: rolling-window Hurst exponent via Anis-Lloyd
    // corrected rescaled-range analysis at dyadic scales.
    py::class_<screamer::RollingHurst, screamer::ScreamerBase>(m, "RollingHurst")
        .def(py::init<int, int, const std::string&>(),
             py::arg("window_size") = 256,
             py::arg("min_scale") = 4,
             py::arg("method") = "rs")
        .def("__call__", &screamer::RollingHurst::operator(), py::arg("value"))
        .def("reset", &screamer::RollingHurst::reset, "Reset.");

    py::class_<screamer::RollingMedian, screamer::ScreamerBase>(m, "RollingMedian")
        .def(py::init<int>(), py::arg("window_size") = 20)
        .def("__call__", &screamer::RollingMedian::operator(), py::arg("value"))
        .def("reset", &screamer::RollingMedian::reset, "Reset to the initial state.");

    py::class_<screamer::RollingQuantile, screamer::ScreamerBase>(m, "RollingQuantile")
        .def(py::init<int, double>(), py::arg("window_size") = 20, py::arg("quantile") = 0.5)
        .def("__call__", &screamer::RollingQuantile::operator(), py::arg("value"))
        .def("reset", &screamer::RollingQuantile::reset, "Reset to the initial state.");

    py::class_<screamer::RollingZscore, screamer::ScreamerBase>(m, "RollingZscore")
        .def(py::init<int, const std::string&>(),
            py::arg("window_size") = 20,
            py::arg("start_policy") = "strict")
        .def("__call__", &screamer::RollingZscore::operator(), py::arg("value"))
        .def("reset", &screamer::RollingZscore::reset, "Reset to the initial state.");

    py::class_<screamer::RollingPoly1, screamer::ScreamerBase>(m, "RollingPoly1")
        .def(py::init<int, int, const std::string&>(),
            py::arg("window_size") = 20,
            py::arg("derivative_order") = 0,
            py::arg("start_policy") = "strict")
        .def("__call__", &screamer::RollingPoly1::operator(), py::arg("value"))
        .def("reset", &screamer::RollingPoly1::reset, "Reset to the initial state.");


    py::class_<screamer::RollingPoly2, screamer::ScreamerBase>(m, "RollingPoly2")
        .def(py::init<int, int, const std::string&>(),
            py::arg("window_size") = 20,
            py::arg("derivative_order") = 0,
            py::arg("start_policy") = "strict")
        .def("__call__", &screamer::RollingPoly2::operator(), py::arg("value"))
        .def("reset", &screamer::RollingPoly2::reset, "Reset to the initial state.");


     py::class_<screamer::RollingSigmaClip, screamer::ScreamerBase>(m, "RollingSigmaClip")
        .def(py::init<int, std::optional<double>, std::optional<double>, std::optional<int>>(),
            py::arg("window_size") = 20,
            py::arg("lower") = std::nullopt,
            py::arg("upper") = std::nullopt,
            py::arg("output") = std::nullopt
        )
        .def("__call__", &screamer::RollingSigmaClip::operator(), py::arg("value"))
        .def("reset", &screamer::RollingSigmaClip::reset, "Reset to the initial state.");

    // Canonical Hampel filter (causal trailing-window): flag samples that are
    // more than n_sigma robust std devs (1.4826 * MAD) from the window median and
    // replace them with that median. output: 0 cleaned, 1 outlier flag, 2 NaN.
    py::class_<screamer::Hampel, screamer::ScreamerBase>(m, "Hampel")
        .def(py::init<int, double, std::optional<int>, const std::string&>(),
            py::arg("window_size") = 20,
            py::arg("n_sigma") = 3.0,
            py::arg("output") = std::nullopt,
            py::arg("start_policy") = "strict")
        .def("__call__", &screamer::Hampel::operator(), py::arg("value"))
        .def("reset", &screamer::Hampel::reset, "Reset to the initial state.");

    // Causal impulse/glitch remover for non-stationary signals: detects spikes on
    // the trailing first difference (trend-free) and replaces them with the window
    // median. output: 0 cleaned, 1 outlier flag, 2 NaN.
    py::class_<screamer::ImpulseClip, screamer::ScreamerBase>(m, "ImpulseClip")
        .def(py::init<int, double, std::optional<int>, const std::string&>(),
            py::arg("window_size") = 20,
            py::arg("n_sigma") = 4.0,
            py::arg("output") = std::nullopt,
            py::arg("start_policy") = "strict")
        .def("__call__", &screamer::ImpulseClip::operator(), py::arg("value"))
        .def("reset", &screamer::ImpulseClip::reset, "Reset to the initial state.");


     py::class_<screamer::RollingOU, screamer::ScreamerBase>(m, "RollingOU")
        .def(py::init<int, std::optional<int>, const std::string&>(),
            py::arg("window_size") = 20,
            py::arg("output") = std::nullopt,
            py::arg("start_policy") = "strict")
        .def("__call__", &screamer::RollingOU::operator(), py::arg("value"))
        .def("reset", &screamer::RollingOU::reset, "Reset to the initial state.");

    py::class_<screamer::RollingRSI, screamer::ScreamerBase>(m, "RollingRSI")
        .def(py::init<int, const std::string&, const std::string&>(),
            py::arg("window_size") = 14,
            py::arg("method") = "wilder",
            py::arg("start_policy") = "strict")
        .def("__call__", &screamer::RollingRSI::operator(), py::arg("value"))
        .def("reset", &screamer::RollingRSI::reset, "Reset to the initial state.");

    // RollingMinMax: 1 input, 2 outputs (min, max). Inherits from
    // FunctorBase<_, 1, 2>, NOT ScreamerBase. The dispatcher returns a
    // tuple per scalar call and an array of shape (..., 2) per batch.
    py::class_<screamer::RollingMinMax, screamer::EvalOp>(m, "RollingMinMax")
        .def(py::init<int>(), py::arg("window_size") = 20)
        .def("__call__", &screamer::RollingMinMax::handle_input)
        .def("reset", &screamer::RollingMinMax::reset, "Reset to the initial state.");

    // BollingerBands: 1 input, 3 outputs (lower, mid, upper).
    // FunctorBase<_, 1, 3>. Per scalar call returns a 3-tuple; per batch
    // returns an array of shape (..., 3).
    py::class_<screamer::BollingerBands, screamer::EvalOp>(m, "BollingerBands")
        .def(py::init<int, double, const std::string&>(),
             py::arg("window_size") = 20,
             py::arg("num_std") = 2.0,
             py::arg("start_policy") = "strict")
        .def("__call__", &screamer::BollingerBands::handle_input)
        .def("reset", &screamer::BollingerBands::reset, "Reset to the initial state.");

}
