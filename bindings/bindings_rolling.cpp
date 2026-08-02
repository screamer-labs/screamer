#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include "screamer/common/base.h"
#include "screamer/common/dispatch.h"
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
#include "screamer/plus_di.h"
#include "screamer/minus_di.h"
#include "screamer/plus_dm.h"
#include "screamer/minus_dm.h"
#include "screamer/dx.h"
#include "screamer/adxr.h"
#include "screamer/vwap.h"
#include "screamer/obv.h"
#include "screamer/ad.h"
#include "screamer/adosc.h"
#include "screamer/mfi.h"
#include "screamer/rolling_tsf.h"
#include "screamer/rolling_rank.h"
#include "screamer/rolling_percentile.h"
#include "screamer/rolling_hurst.h"
#include <string>

namespace nb = nanobind;
using namespace nb::literals;

void init_bindings_rolling(nb::module_& m) {

    nb::class_<screamer::RollingMean, screamer::ScreamerBase>(m, "RollingMean")
        .def(nb::init<int, const std::string&>(), 
            "window_size"_a = 20, 
            "start_policy"_a = "strict")
        .def("__call__", &screamer::screamer_call, "value"_a)
        .def("reset", &screamer::RollingMean::reset, "Reset to the initial state.");

    nb::class_<screamer::RollingRms, screamer::ScreamerBase>(m, "RollingRms")
        .def(nb::init<int, const std::string&>(),
            "window_size"_a = 20,
            "start_policy"_a = "strict")
        .def("__call__", &screamer::screamer_call, "value"_a)
        .def("reset", &screamer::RollingRms::reset, "Reset to the initial state.");

    nb::class_<screamer::RollingSum, screamer::ScreamerBase>(m, "RollingSum")
        .def(nb::init<int, const std::string&>(), 
            "window_size"_a = 20, 
            "start_policy"_a = "strict")
        .def("__call__", &screamer::screamer_call, "value"_a)
        .def("reset", &screamer::RollingSum::reset, "Reset to the initial state.");

    nb::class_<screamer::RollingStd, screamer::ScreamerBase>(m, "RollingStd")
        .def(nb::init<int, const std::string&>(),
            "window_size"_a = 20,
            "start_policy"_a = "strict")
        .def("__call__", &screamer::screamer_call, "value"_a)
        .def("reset", &screamer::RollingStd::reset, "Reset to the initial state.");

    nb::class_<screamer::RollingVar, screamer::ScreamerBase>(m, "RollingVar")
        .def(nb::init<int, const std::string&>(),
            "window_size"_a = 20,
            "start_policy"_a = "strict")
        .def("__call__", &screamer::screamer_call, "value"_a)
        .def("reset", &screamer::RollingVar::reset, "Reset to the initial state.");

    nb::class_<screamer::RollingSkew, screamer::ScreamerBase>(m, "RollingSkew")
        .def(nb::init<int, const std::string&>(),
            "window_size"_a = 20,
            "start_policy"_a = "strict")
        .def("__call__", &screamer::screamer_call, "value"_a)
        .def("reset", &screamer::RollingSkew::reset, "Reset to the initial state.");

    nb::class_<screamer::RollingKurt, screamer::ScreamerBase>(m, "RollingKurt")
        .def(nb::init<int, const std::string&>(),
            "window_size"_a = 20,
            "start_policy"_a = "strict")
        .def("__call__", &screamer::screamer_call, "value"_a)
        .def("reset", &screamer::RollingKurt::reset, "Reset to the initial state.");

    nb::class_<screamer::RollingMin, screamer::ScreamerBase>(m, "RollingMin")
        .def(nb::init<int>(), "window_size"_a = 20)
        .def("__call__", &screamer::screamer_call, "value"_a)
        .def("reset", &screamer::RollingMin::reset, "Reset to the initial state.");

    nb::class_<screamer::RollingMax, screamer::ScreamerBase>(m, "RollingMax")
        .def(nb::init<int>(), "window_size"_a = 20)
        .def("__call__", &screamer::screamer_call, "value"_a)
        .def("reset", &screamer::RollingMax::reset, "Reset to the initial state.");

    // Position of the rolling minimum / maximum within the window.
    // 0 = oldest sample, window_size-1 = newest. Same monotonic-deque
    // primitive as RollingMin / RollingMax, exposed via the front
    // element's window offset.
    nb::class_<screamer::RollingArgmin, screamer::ScreamerBase>(m, "RollingArgmin")
        .def(nb::init<int>(), "window_size"_a = 20)
        .def("__call__", &screamer::screamer_call, "value"_a)
        .def("reset", &screamer::RollingArgmin::reset, "Reset to the initial state.");

    nb::class_<screamer::RollingArgmax, screamer::ScreamerBase>(m, "RollingArgmax")
        .def(nb::init<int>(), "window_size"_a = 20)
        .def("__call__", &screamer::screamer_call, "value"_a)
        .def("reset", &screamer::RollingArgmax::reset, "Reset to the initial state.");

    // RollingRange: max - min. Two monotonic deques internally,
    // composed at the primitive level (same algorithm RollingMinMax
    // runs, returned as a single scalar instead of a tuple).
    nb::class_<screamer::RollingRange, screamer::ScreamerBase>(m, "RollingRange")
        .def(nb::init<int>(), "window_size"_a = 20)
        .def("__call__", &screamer::screamer_call, "value"_a)
        .def("reset", &screamer::RollingRange::reset, "Reset to the initial state.");

    // Mean absolute deviation, mean(|x - rolling_mean|). O(W) per step
    // (provably no closed-form O(1) exists; the moving mean
    // re-evaluates all W abs-deviations each step).
    nb::class_<screamer::RollingMad, screamer::ScreamerBase>(m, "RollingMad")
        .def(nb::init<int, const std::string&>(),
            "window_size"_a = 20,
            "start_policy"_a = "strict")
        .def("__call__", &screamer::screamer_call, "value"_a)
        .def("reset", &screamer::RollingMad::reset, "Reset to the initial state.");

    // Robust scale: the median absolute deviation, median(|x - median|), over the
    // trailing window. Unlike RollingMad (mean absolute deviation) it is robust to
    // outliers, and it is the scale primitive behind Hampel and ImpulseClip.
    nb::class_<screamer::RollingMedianAD, screamer::ScreamerBase>(m, "RollingMedianAD")
        .def(nb::init<int, const std::string&>(),
            "window_size"_a = 20,
            "start_policy"_a = "strict")
        .def("__call__", &screamer::screamer_call, "value"_a)
        .def("reset", &screamer::RollingMedianAD::reset, "Reset to the initial state.");

    // Inter-quartile range = q75 - q25. Single shared OST queried
    // twice (vs. two RollingQuantile instances which would use two
    // independent trees). Same O(log W) per step, half the memory
    // and inserts.
    nb::class_<screamer::RollingIqr, screamer::ScreamerBase>(m, "RollingIqr")
        .def(nb::init<int>(), "window_size"_a = 20)
        .def("__call__", &screamer::screamer_call, "value"_a)
        .def("reset", &screamer::RollingIqr::reset, "Reset to the initial state.");

    // WMA: linearly-weighted moving average. O(1) per step via the
    // identity W[t] - W[t-1] = w*x[t] - S[t-1] where S is the simple
    // rolling sum of the previous window.
    nb::class_<screamer::WMA, screamer::ScreamerBase>(m, "WMA")
        .def(nb::init<int, const std::string&>(),
            "window_size"_a = 20,
            "start_policy"_a = "strict")
        .def("__call__", &screamer::screamer_call, "value"_a)
        .def("reset", &screamer::WMA::reset, "Reset to the initial state.");

    // DEMA / TEMA: double / triple exponential MA (Mulloy 1994). Pure
    // composition of 2 / 3 chained EwMean instances.
    nb::class_<screamer::DEMA, screamer::ScreamerBase>(m, "DEMA")
        .def(
          nb::init<
               std::optional<double>,
               std::optional<double>,
               std::optional<double>,
               std::optional<double>
          >(),
          "com"_a = nb::none(),
          "span"_a = nb::none(),
          "halflife"_a = nb::none(),
          "alpha"_a = nb::none()
        )
        .def("__call__", &screamer::screamer_call, "value"_a)
        .def("reset", &screamer::DEMA::reset, "Reset to the initial state.");

    nb::class_<screamer::TEMA, screamer::ScreamerBase>(m, "TEMA")
        .def(
          nb::init<
               std::optional<double>,
               std::optional<double>,
               std::optional<double>,
               std::optional<double>
          >(),
          "com"_a = nb::none(),
          "span"_a = nb::none(),
          "halflife"_a = nb::none(),
          "alpha"_a = nb::none()
        )
        .def("__call__", &screamer::screamer_call, "value"_a)
        .def("reset", &screamer::TEMA::reset, "Reset to the initial state.");

    // TRIMA: triangular MA, SMA(SMA(x)). Pure composition of two
    // detail::RollingMean instances. Strict warmup enforced by counter.
    nb::class_<screamer::TRIMA, screamer::ScreamerBase>(m, "TRIMA")
        .def(nb::init<int>(), "window_size"_a = 20)
        .def("__call__", &screamer::screamer_call, "value"_a)
        .def("reset", &screamer::TRIMA::reset, "Reset to the initial state.");

    // HullMA: WMA(2*WMA(x, n/2) - WMA(x, n), sqrt(n)). Pure composition
    // of three WMA instances. Inner WMAs use "expanding" so they don't
    // emit NaN; HullMA enforces strict warmup itself.
    nb::class_<screamer::HullMA, screamer::ScreamerBase>(m, "HullMA")
        .def(nb::init<int>(), "window_size"_a = 20)
        .def("__call__", &screamer::screamer_call, "value"_a)
        .def("reset", &screamer::HullMA::reset, "Reset to the initial state.");

    // KAMA: Kaufman's Adaptive MA. Smoothing constant adapts to the
    // efficiency ratio (net displacement / total absolute travel).
    nb::class_<screamer::KAMA, screamer::ScreamerBase>(m, "KAMA")
        .def(nb::init<int, int, int>(),
            "window_size"_a = 10,
            "fast"_a = 2,
            "slow"_a = 30)
        .def("__call__", &screamer::screamer_call, "value"_a)
        .def("reset", &screamer::KAMA::reset, "Reset to the initial state.");

    // MACD: (macd, signal, histogram). 1->3 functor composing three
    // pandas adjust=True EMAs (our EwMean).
    nb::class_<screamer::MACD, screamer::EvalOp>(m, "MACD")
        .def(nb::init<int, int, int>(),
            "fast"_a = 12,
            "slow"_a = 26,
            "signal"_a = 9)
        .def("__call__", &screamer::MACD::handle_input)
        .def("reset", &screamer::MACD::reset, "Reset to the initial state.");

    // WilliamsR: 3->1, takes (high, low, close), returns %R in [-100, 0].
    nb::class_<screamer::WilliamsR, screamer::EvalOp>(m, "WilliamsR")
        .def(nb::init<int>(), "window_size"_a = 14)
        .def("__call__", &screamer::WilliamsR::handle_input)
        .def("reset", &screamer::WilliamsR::reset, "Reset to the initial state.");

    // Stoch: 3->2, takes (high, low, close), returns (%K, %D). With
    // smooth_k=1 this is the "fast" stochastic (Lane's original);
    // with smooth_k>=2 it is the "slow" stochastic (talib.STOCH).
    nb::class_<screamer::Stoch, screamer::EvalOp>(m, "Stoch")
        .def(nb::init<int, int, int>(),
            "fastk_period"_a = 14,
            "smooth_k"_a = 3,
            "d"_a = 3)
        .def("__call__", &screamer::Stoch::handle_input)
        .def("reset", &screamer::Stoch::reset, "Reset to the initial state.");

    // TRIX: 100 * 1-period ROC of triple-smoothed EMA. Composes
    // three EwMean instances and tracks the previous ema3 for the
    // final ratio.
    nb::class_<screamer::TRIX, screamer::ScreamerBase>(m, "TRIX")
        .def(nb::init<int>(), "span"_a = 14)
        .def("__call__", &screamer::screamer_call, "value"_a)
        .def("reset", &screamer::TRIX::reset, "Reset to the initial state.");

    // BOP: Balance of Power. 4 -> 1 on (open, high, low, close).
    nb::class_<screamer::BOP, screamer::EvalOp>(m, "BOP")
        .def(nb::init<>())
        .def("__call__", &screamer::BOP::handle_input)
        .def("reset", &screamer::BOP::reset, "Reset to the initial state.");

    // CCI: Commodity Channel Index. 3 -> 1 on (high, low, close).
    nb::class_<screamer::CCI, screamer::EvalOp>(m, "CCI")
        .def(nb::init<int>(), "window_size"_a = 14)
        .def("__call__", &screamer::CCI::handle_input)
        .def("reset", &screamer::CCI::reset, "Reset to the initial state.");

    // UltimateOscillator: 3 -> 1 on (high, low, close); weighted
    // average over three timeframes.
    nb::class_<screamer::UltimateOscillator, screamer::EvalOp>(m, "UltimateOscillator")
        .def(nb::init<int, int, int>(),
            "period1"_a = 7,
            "period2"_a = 14,
            "period3"_a = 28)
        .def("__call__", &screamer::UltimateOscillator::handle_input)
        .def("reset", &screamer::UltimateOscillator::reset,
             "Reset to the initial state.");

    // StochRSI: 1 -> 2; Stochastic of RSI. Default smooth_k=1 (fast,
    // matching TA-Lib's STOCHRSI); set smooth_k >= 2 for slow form.
    nb::class_<screamer::StochRSI, screamer::EvalOp>(m, "StochRSI")
        .def(nb::init<int, int, int, int>(),
            "rsi_period"_a = 14,
            "stoch_period"_a = 14,
            "smooth_k"_a = 1,
            "d"_a = 3)
        .def("__call__", &screamer::StochRSI::handle_input)
        .def("reset", &screamer::StochRSI::reset, "Reset to the initial state.");

    // ----- Range-based volatility estimators -----
    // Each estimator has a *Var (variance) form and a *Vol (= sqrt of
    // *Var) form, in both Rolling and EW smoothing variants.

    // Parkinson (1980): H, L. 2 -> 1.
    nb::class_<screamer::RollingParkinsonVar, screamer::EvalOp>(m, "RollingParkinsonVar")
        .def(nb::init<int>(), "window_size"_a = 20)
        .def("__call__", &screamer::RollingParkinsonVar::handle_input)
        .def("reset", &screamer::RollingParkinsonVar::reset, "Reset.");

    nb::class_<screamer::RollingParkinsonVol, screamer::EvalOp>(m, "RollingParkinsonVol")
        .def(nb::init<int>(), "window_size"_a = 20)
        .def("__call__", &screamer::RollingParkinsonVol::handle_input)
        .def("reset", &screamer::RollingParkinsonVol::reset, "Reset.");

    nb::class_<screamer::EwParkinsonVar, screamer::EvalOp>(m, "EwParkinsonVar")
        .def(nb::init<std::optional<double>, std::optional<double>,
                       std::optional<double>, std::optional<double>>(),
             "com"_a = nb::none(), "span"_a = nb::none(),
             "halflife"_a = nb::none(), "alpha"_a = nb::none())
        .def("__call__", &screamer::EwParkinsonVar::handle_input)
        .def("reset", &screamer::EwParkinsonVar::reset, "Reset.");

    nb::class_<screamer::EwParkinsonVol, screamer::EvalOp>(m, "EwParkinsonVol")
        .def(nb::init<std::optional<double>, std::optional<double>,
                       std::optional<double>, std::optional<double>>(),
             "com"_a = nb::none(), "span"_a = nb::none(),
             "halflife"_a = nb::none(), "alpha"_a = nb::none())
        .def("__call__", &screamer::EwParkinsonVol::handle_input)
        .def("reset", &screamer::EwParkinsonVol::reset, "Reset.");

    // Garman-Klass (1980): O, H, L, C. 4 -> 1.
    nb::class_<screamer::RollingGarmanKlassVar, screamer::EvalOp>(m, "RollingGarmanKlassVar")
        .def(nb::init<int>(), "window_size"_a = 20)
        .def("__call__", &screamer::RollingGarmanKlassVar::handle_input)
        .def("reset", &screamer::RollingGarmanKlassVar::reset, "Reset.");

    nb::class_<screamer::RollingGarmanKlassVol, screamer::EvalOp>(m, "RollingGarmanKlassVol")
        .def(nb::init<int>(), "window_size"_a = 20)
        .def("__call__", &screamer::RollingGarmanKlassVol::handle_input)
        .def("reset", &screamer::RollingGarmanKlassVol::reset, "Reset.");

    nb::class_<screamer::EwGarmanKlassVar, screamer::EvalOp>(m, "EwGarmanKlassVar")
        .def(nb::init<std::optional<double>, std::optional<double>,
                       std::optional<double>, std::optional<double>>(),
             "com"_a = nb::none(), "span"_a = nb::none(),
             "halflife"_a = nb::none(), "alpha"_a = nb::none())
        .def("__call__", &screamer::EwGarmanKlassVar::handle_input)
        .def("reset", &screamer::EwGarmanKlassVar::reset, "Reset.");

    nb::class_<screamer::EwGarmanKlassVol, screamer::EvalOp>(m, "EwGarmanKlassVol")
        .def(nb::init<std::optional<double>, std::optional<double>,
                       std::optional<double>, std::optional<double>>(),
             "com"_a = nb::none(), "span"_a = nb::none(),
             "halflife"_a = nb::none(), "alpha"_a = nb::none())
        .def("__call__", &screamer::EwGarmanKlassVol::handle_input)
        .def("reset", &screamer::EwGarmanKlassVol::reset, "Reset.");

    // Rogers-Satchell (1991): O, H, L, C; drift-robust. 4 -> 1.
    nb::class_<screamer::RollingRogersSatchellVar, screamer::EvalOp>(m, "RollingRogersSatchellVar")
        .def(nb::init<int>(), "window_size"_a = 20)
        .def("__call__", &screamer::RollingRogersSatchellVar::handle_input)
        .def("reset", &screamer::RollingRogersSatchellVar::reset, "Reset.");

    nb::class_<screamer::RollingRogersSatchellVol, screamer::EvalOp>(m, "RollingRogersSatchellVol")
        .def(nb::init<int>(), "window_size"_a = 20)
        .def("__call__", &screamer::RollingRogersSatchellVol::handle_input)
        .def("reset", &screamer::RollingRogersSatchellVol::reset, "Reset.");

    nb::class_<screamer::EwRogersSatchellVar, screamer::EvalOp>(m, "EwRogersSatchellVar")
        .def(nb::init<std::optional<double>, std::optional<double>,
                       std::optional<double>, std::optional<double>>(),
             "com"_a = nb::none(), "span"_a = nb::none(),
             "halflife"_a = nb::none(), "alpha"_a = nb::none())
        .def("__call__", &screamer::EwRogersSatchellVar::handle_input)
        .def("reset", &screamer::EwRogersSatchellVar::reset, "Reset.");

    nb::class_<screamer::EwRogersSatchellVol, screamer::EvalOp>(m, "EwRogersSatchellVol")
        .def(nb::init<std::optional<double>, std::optional<double>,
                       std::optional<double>, std::optional<double>>(),
             "com"_a = nb::none(), "span"_a = nb::none(),
             "halflife"_a = nb::none(), "alpha"_a = nb::none())
        .def("__call__", &screamer::EwRogersSatchellVol::handle_input)
        .def("reset", &screamer::EwRogersSatchellVol::reset, "Reset.");

    // TrueRange / ATR / NATR (Wilder family). 3 -> 1 on (high, low, close).
    nb::class_<screamer::TrueRange, screamer::EvalOp>(m, "TrueRange")
        .def(nb::init<>())
        .def("__call__", &screamer::TrueRange::handle_input)
        .def("reset", &screamer::TrueRange::reset, "Reset to the initial state.");

    nb::class_<screamer::ATR, screamer::EvalOp>(m, "ATR")
        .def(nb::init<int>(), "window_size"_a = 14)
        .def("__call__", &screamer::ATR::handle_input)
        .def("reset", &screamer::ATR::reset, "Reset to the initial state.");

    nb::class_<screamer::NATR, screamer::EvalOp>(m, "NATR")
        .def(nb::init<int>(), "window_size"_a = 14)
        .def("__call__", &screamer::NATR::handle_input)
        .def("reset", &screamer::NATR::reset, "Reset to the initial state.");

    // Donchian / Keltner channels (envelope-style indicators).
    nb::class_<screamer::DonchianChannels, screamer::EvalOp>(m, "DonchianChannels")
        .def(nb::init<int>(), "window_size"_a = 20)
        .def("__call__", &screamer::DonchianChannels::handle_input)
        .def("reset", &screamer::DonchianChannels::reset, "Reset.");

    nb::class_<screamer::KeltnerChannels, screamer::EvalOp>(m, "KeltnerChannels")
        .def(nb::init<int, double>(),
             "window_size"_a = 20, "num_atr"_a = 2.0)
        .def("__call__", &screamer::KeltnerChannels::handle_input)
        .def("reset", &screamer::KeltnerChannels::reset, "Reset.");

    // Yang-Zhang volatility (the most efficient classical range-based
    // estimator; handles both drift and overnight gaps).
    nb::class_<screamer::RollingYangZhangVar, screamer::EvalOp>(m, "RollingYangZhangVar")
        .def(nb::init<int>(), "window_size"_a = 20)
        .def("__call__", &screamer::RollingYangZhangVar::handle_input)
        .def("reset", &screamer::RollingYangZhangVar::reset, "Reset.");

    nb::class_<screamer::RollingYangZhangVol, screamer::EvalOp>(m, "RollingYangZhangVol")
        .def(nb::init<int>(), "window_size"_a = 20)
        .def("__call__", &screamer::RollingYangZhangVol::handle_input)
        .def("reset", &screamer::RollingYangZhangVol::reset, "Reset.");

    // ADX (Wilder, 1978). 3 -> 3 on (high, low, close) returning
    // (plus_di, minus_di, adx). Match talib.PLUS_DI / MINUS_DI / ADX.
    nb::class_<screamer::ADX, screamer::EvalOp>(m, "ADX")
        .def(nb::init<int>(), "window_size"_a = 14)
        .def("__call__", &screamer::ADX::handle_input)
        .def("reset", &screamer::ADX::reset, "Reset to the initial state.");

    // PlusDI / MinusDI: standalone +DI / -DI, sharing DmiCore with ADX.
    nb::class_<screamer::PlusDI, screamer::EvalOp>(m, "PlusDI")
        .def(nb::init<int>(), "window_size"_a = 14)
        .def("__call__", &screamer::PlusDI::handle_input)
        .def("reset", &screamer::PlusDI::reset, "Reset to the initial state.");

    nb::class_<screamer::MinusDI, screamer::EvalOp>(m, "MinusDI")
        .def(nb::init<int>(), "window_size"_a = 14)
        .def("__call__", &screamer::MinusDI::handle_input)
        .def("reset", &screamer::MinusDI::reset, "Reset to the initial state.");

    // PlusDM / MinusDM: standalone +DM / -DM, sharing DmiCore with ADX.
    // High/low only; the node feeds close = high internally.
    nb::class_<screamer::PlusDM, screamer::EvalOp>(m, "PlusDM")
        .def(nb::init<int>(), "window_size"_a = 14)
        .def("__call__", &screamer::PlusDM::handle_input)
        .def("reset", &screamer::PlusDM::reset, "Reset to the initial state.");

    nb::class_<screamer::MinusDM, screamer::EvalOp>(m, "MinusDM")
        .def(nb::init<int>(), "window_size"_a = 14)
        .def("__call__", &screamer::MinusDM::handle_input)
        .def("reset", &screamer::MinusDM::reset, "Reset to the initial state.");

    // DX: standalone directional index, sharing DmiCore with ADX.
    nb::class_<screamer::DX, screamer::EvalOp>(m, "DX")
        .def(nb::init<int>(), "window_size"_a = 14)
        .def("__call__", &screamer::DX::handle_input)
        .def("reset", &screamer::DX::reset, "Reset to the initial state.");

    // ADXR: mean of ADX now and ADX (window_size - 1) bars ago, sharing
    // DmiCore with ADX.
    nb::class_<screamer::ADXR, screamer::EvalOp>(m, "ADXR")
        .def(nb::init<int>(), "window_size"_a = 14)
        .def("__call__", &screamer::ADXR::handle_input)
        .def("reset", &screamer::ADXR::reset, "Reset to the initial state.");

    // Volume-aware indicators.
    nb::class_<screamer::RollingVWAP, screamer::EvalOp>(m, "RollingVWAP")
        .def(nb::init<int>(), "window_size"_a = 20)
        .def("__call__", &screamer::RollingVWAP::handle_input)
        .def("reset", &screamer::RollingVWAP::reset, "Reset.");

    nb::class_<screamer::OBV, screamer::EvalOp>(m, "OBV")
        .def(nb::init<>())
        .def("__call__", &screamer::OBV::handle_input)
        .def("reset", &screamer::OBV::reset, "Reset.");

    nb::class_<screamer::AD, screamer::EvalOp>(m, "AD")
        .def(nb::init<>())
        .def("__call__", &screamer::AD::handle_input)
        .def("reset", &screamer::AD::reset, "Reset.");

    nb::class_<screamer::ADOSC, screamer::EvalOp>(m, "ADOSC")
        .def(nb::init<int, int>(), "fast"_a = 3, "slow"_a = 10)
        .def("__call__", &screamer::ADOSC::handle_input)
        .def("reset", &screamer::ADOSC::reset, "Reset.");

    nb::class_<screamer::MFI, screamer::EvalOp>(m, "MFI")
        .def(nb::init<int>(), "window_size"_a = 14)
        .def("__call__", &screamer::MFI::handle_input)
        .def("reset", &screamer::MFI::reset, "Reset.");

    // Time-Series Forecast (TA-Lib's TSF): linear regression vs
    // time projected one step ahead. Composes detail::RollingSum.
    nb::class_<screamer::RollingTSF, screamer::ScreamerBase>(m, "RollingTSF")
        .def(nb::init<int>(), "window_size"_a = 20)
        .def("__call__", &screamer::screamer_call, "value"_a)
        .def("reset", &screamer::RollingTSF::reset, "Reset.");

    // RollingRank / RollingPercentile: position of latest value in
    // the trailing window. pandas-style "average" tie rule.
    nb::class_<screamer::RollingRank, screamer::ScreamerBase>(m, "RollingRank")
        .def(nb::init<int>(), "window_size"_a = 20)
        .def("__call__", &screamer::screamer_call, "value"_a)
        .def("reset", &screamer::RollingRank::reset, "Reset.");

    nb::class_<screamer::RollingPercentile, screamer::ScreamerBase>(m, "RollingPercentile")
        .def(nb::init<int>(), "window_size"_a = 20)
        .def("__call__", &screamer::screamer_call, "value"_a)
        .def("reset", &screamer::RollingPercentile::reset, "Reset.");

    // RollingHurst: rolling-window Hurst exponent via Anis-Lloyd
    // corrected rescaled-range analysis at dyadic scales.
    nb::class_<screamer::RollingHurst, screamer::ScreamerBase>(m, "RollingHurst")
        .def(nb::init<int, int, const std::string&>(),
             "window_size"_a = 256,
             "min_scale"_a = 4,
             "method"_a = "rs")
        .def("__call__", &screamer::screamer_call, "value"_a)
        .def("reset", &screamer::RollingHurst::reset, "Reset.");

    nb::class_<screamer::RollingMedian, screamer::ScreamerBase>(m, "RollingMedian")
        .def(nb::init<int>(), "window_size"_a = 20)
        .def("__call__", &screamer::screamer_call, "value"_a)
        .def("reset", &screamer::RollingMedian::reset, "Reset to the initial state.");

    nb::class_<screamer::RollingQuantile, screamer::ScreamerBase>(m, "RollingQuantile")
        .def(nb::init<int, double>(), "window_size"_a = 20, "quantile"_a = 0.5)
        .def("__call__", &screamer::screamer_call, "value"_a)
        .def("reset", &screamer::RollingQuantile::reset, "Reset to the initial state.");

    nb::class_<screamer::RollingZscore, screamer::ScreamerBase>(m, "RollingZscore")
        .def(nb::init<int, const std::string&>(),
            "window_size"_a = 20,
            "start_policy"_a = "strict")
        .def("__call__", &screamer::screamer_call, "value"_a)
        .def("reset", &screamer::RollingZscore::reset, "Reset to the initial state.");

    nb::class_<screamer::RollingPoly1, screamer::ScreamerBase>(m, "RollingPoly1")
        .def(nb::init<int, int, const std::string&>(),
            "window_size"_a = 20,
            "derivative_order"_a = 0,
            "start_policy"_a = "strict")
        .def("__call__", &screamer::screamer_call, "value"_a)
        .def("reset", &screamer::RollingPoly1::reset, "Reset to the initial state.");


    nb::class_<screamer::RollingPoly2, screamer::ScreamerBase>(m, "RollingPoly2")
        .def(nb::init<int, int, const std::string&>(),
            "window_size"_a = 20,
            "derivative_order"_a = 0,
            "start_policy"_a = "strict")
        .def("__call__", &screamer::screamer_call, "value"_a)
        .def("reset", &screamer::RollingPoly2::reset, "Reset to the initial state.");


     nb::class_<screamer::RollingSigmaClip, screamer::ScreamerBase>(m, "RollingSigmaClip")
        .def(nb::init<int, std::optional<double>, std::optional<double>, const std::string&, const std::string&>(),
            "window_size"_a = 20,
            "lower"_a = nb::none(),
            "upper"_a = nb::none(),
            "output"_a = "clipped",
            "start_policy"_a = "strict"
        )
        .def("__call__", &screamer::screamer_call, "value"_a)
        .def("reset", &screamer::RollingSigmaClip::reset, "Reset to the initial state.");

    // Canonical Hampel filter (causal trailing-window): flag samples that are
    // more than n_sigma robust std devs (1.4826 * MAD) from the window median and
    // replace them with that median. output: "cleaned", "flag", or "nan".
    nb::class_<screamer::Hampel, screamer::ScreamerBase>(m, "Hampel")
        .def(nb::init<int, double, const std::string&, const std::string&>(),
            "window_size"_a = 20,
            "n_sigma"_a = 3.0,
            "output"_a = "cleaned",
            "start_policy"_a = "strict")
        .def("__call__", &screamer::screamer_call, "value"_a)
        .def("reset", &screamer::Hampel::reset, "Reset to the initial state.");

    // Causal impulse/glitch remover for non-stationary signals: detects spikes on
    // the trailing first difference (trend-free) and replaces them with the window
    // median. output: "cleaned", "flag", or "nan".
    nb::class_<screamer::ImpulseClip, screamer::ScreamerBase>(m, "ImpulseClip")
        .def(nb::init<int, double, const std::string&, const std::string&>(),
            "window_size"_a = 20,
            "n_sigma"_a = 4.0,
            "output"_a = "cleaned",
            "start_policy"_a = "strict")
        .def("__call__", &screamer::screamer_call, "value"_a)
        .def("reset", &screamer::ImpulseClip::reset, "Reset to the initial state.");


     nb::class_<screamer::RollingOU, screamer::ScreamerBase>(m, "RollingOU")
        .def(nb::init<int, const std::string&, const std::string&>(),
            "window_size"_a = 20,
            "output"_a = "mrr",
            "start_policy"_a = "strict")
        .def("__call__", &screamer::screamer_call, "value"_a)
        .def("reset", &screamer::RollingOU::reset, "Reset to the initial state.");

    nb::class_<screamer::RollingRSI, screamer::ScreamerBase>(m, "RollingRSI")
        .def(nb::init<int, const std::string&, const std::string&>(),
            "window_size"_a = 14,
            "method"_a = "wilder",
            "start_policy"_a = "strict")
        .def("__call__", &screamer::screamer_call, "value"_a)
        .def("reset", &screamer::RollingRSI::reset, "Reset to the initial state.");

    // RollingMinMax: 1 input, 2 outputs (min, max). Inherits from
    // FunctorBase<_, 1, 2>, NOT ScreamerBase. The dispatcher returns a
    // tuple per scalar call and an array of shape (..., 2) per batch.
    nb::class_<screamer::RollingMinMax, screamer::EvalOp>(m, "RollingMinMax")
        .def(nb::init<int>(), "window_size"_a = 20)
        .def("__call__", &screamer::RollingMinMax::handle_input)
        .def("reset", &screamer::RollingMinMax::reset, "Reset to the initial state.");

    // BollingerBands: 1 input, 3 outputs (lower, mid, upper).
    // FunctorBase<_, 1, 3>. Per scalar call returns a 3-tuple; per batch
    // returns an array of shape (..., 3).
    nb::class_<screamer::BollingerBands, screamer::EvalOp>(m, "BollingerBands")
        .def(nb::init<int, double, const std::string&>(),
             "window_size"_a = 20,
             "num_std"_a = 2.0,
             "start_policy"_a = "strict")
        .def("__call__", &screamer::BollingerBands::handle_input)
        .def("reset", &screamer::BollingerBands::reset, "Reset to the initial state.");

}
