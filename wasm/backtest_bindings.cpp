// Embind bindings for backtestReport (js/src/backtest.ts). The report is
// computed by include/screamer/backtest_summary.h, as Python's backtest_report
// computes it; JavaScript only labels the columns and the summary with the
// names backtestReportNames gives. Buffers are WASM heap addresses from
// allocF64.
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>
#include <emscripten/bind.h>
#include <emscripten/val.h>
#include "screamer/backtest_summary.h"

namespace {

constexpr std::size_t kEngine = screamer::kEngineColumns.size();
constexpr std::size_t kReport = screamer::BacktestReport::column_names.size();
constexpr std::size_t kStatsCols = screamer::BacktestStats::column_names.size();

// The engine output `eng` (T rows of `ncols` values, row-major) as the
// running columns, column-major in `out` (4 + 6 + 9 columns of T values, in
// the order of backtestReportNames), and the summary in `summary`
// (kNumStats values). The shape and periods_per_year are checked first.
void backtestReportInto(std::uintptr_t eng, std::size_t T, std::size_t ndim, std::size_t ncols,
                        double periods_per_year, std::uintptr_t out, std::uintptr_t summary) {
    screamer::check_engine_rows(ndim, ncols);
    screamer::BacktestReport rep(periods_per_year);
    screamer::BacktestStats st(periods_per_year);
    const double* e = reinterpret_cast<const double*>(eng);
    std::vector<double> report(T * kReport), stats(T * kStatsCols);
    screamer::backtest_report(rep, st, e, T, static_cast<std::ptrdiff_t>(kEngine), 1,
                              report.data(), stats.data(), reinterpret_cast<double*>(summary));
    double* o = reinterpret_cast<double*>(out);
    const auto columns = [&](const double* rows, std::size_t width) {
        for (std::size_t c = 0; c < width; ++c, o += T)
            for (std::size_t t = 0; t < T; ++t) o[t] = rows[t * width + c];
    };
    columns(e, kEngine);
    columns(report.data(), kReport);
    columns(stats.data(), kStatsCols);
}

// [engine columns, BacktestReport columns, BacktestStats columns, summary].
emscripten::val backtestReportNames() {
    const auto names = [](const auto& a) {
        emscripten::val out = emscripten::val::array();
        for (std::size_t i = 0; i < a.size(); ++i) out.set(i, std::string(a[i]));
        return out;
    };
    emscripten::val summary = emscripten::val::array();
    for (std::size_t q = 0; q < screamer::kNumStats; ++q)
        summary.set(q, std::string(screamer::kStats[q].name));
    emscripten::val all = emscripten::val::array();
    all.set(0, names(screamer::kEngineColumns));
    all.set(1, names(screamer::BacktestReport::column_names));
    all.set(2, names(screamer::BacktestStats::column_names));
    all.set(3, summary);
    return all;
}

}  // namespace

EMSCRIPTEN_BINDINGS(screamer_backtest) {
    emscripten::function("backtestReportInto", &backtestReportInto);
    emscripten::function("backtestReportNames", &backtestReportNames);
}
