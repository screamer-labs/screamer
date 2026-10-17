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
#include "screamer/backtest_trade_list.h"

namespace {

constexpr std::size_t kEngine = screamer::kEngineColumns.size();
constexpr std::size_t kReport = screamer::BacktestReport::column_names.size();
constexpr std::size_t kStatsCols = screamer::BacktestStats::column_names.size();

// The shape rule of an engine's output, checked first, so a bad call raises
// the same error class as in Python whatever else is wrong with it.
void checkEngineRows(std::size_t ndim, std::size_t ncols) {
    screamer::check_engine_rows(ndim, ncols);
}

// The engine output `eng` (T rows of `ncols` values, row-major) as the
// computed running columns, column-major in `out` (6 + 9 columns of T values,
// in the order of backtestReportNames), and the summary in `summary`
// (kNumStats values). The engine columns stay in the caller's input. The
// shape and periods_per_year are checked first.
void backtestReportInto(std::uintptr_t eng, std::size_t T, std::size_t ndim, std::size_t ncols,
                        double periods_per_year, std::uintptr_t out, std::uintptr_t summary) {
    screamer::check_engine_rows(ndim, ncols);
    screamer::BacktestReport rep(periods_per_year);
    screamer::BacktestStats st(periods_per_year);
    std::vector<double> report(T * kReport), stats(T * kStatsCols);
    screamer::backtest_report(rep, st, reinterpret_cast<const double*>(eng), T,
                              static_cast<std::ptrdiff_t>(kEngine), 1, report.data(),
                              stats.data(), reinterpret_cast<double*>(summary));
    double* o = reinterpret_cast<double*>(out);
    const auto columns = [&](const double* rows, std::size_t width) {
        for (std::size_t c = 0; c < width; ++c, o += T)
            for (std::size_t t = 0; t < T; ++t) o[t] = rows[t * width + c];
    };
    columns(report.data(), kReport);
    columns(stats.data(), kStatsCols);
}

// backtest_report(by=): the statistics of `groups` groups of the engine
// output `eng` (T rows of `ncols` values, row-major) for the `n_codes` group
// codes at `codes` (doubles holding integers in [0, groups)), written row by
// row to `out` (groups x kGroupStats). The shape and the code count are
// checked first.
void backtestGroupsInto(std::uintptr_t eng, std::size_t T, std::size_t ndim, std::size_t ncols,
                        std::uintptr_t codes, std::size_t n_codes, std::size_t groups,
                        double periods_per_year, std::uintptr_t out) {
    screamer::check_engine_rows(ndim, ncols);
    screamer::check_group_codes(n_codes, T);
    const double* c = reinterpret_cast<const double*>(codes);
    std::vector<std::int64_t> ic(T);
    for (std::size_t t = 0; t < T; ++t) ic[t] = static_cast<std::int64_t>(c[t]);
    screamer::backtest_groups(reinterpret_cast<const double*>(eng), T,
                              static_cast<std::ptrdiff_t>(ncols), 1, ic.data(), groups,
                              periods_per_year, reinterpret_cast<double*>(out));
}

// [engine columns, BacktestReport columns, BacktestStats columns, summary,
// group statistics].
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
    all.set(4, names(screamer::kGroupStats));
    return all;
}

// The round trips of the engine output `eng` (T rows of `ncols` values,
// row-major) as one Float64Array of 7 n values: entry, exit, side, bars, pnl,
// cost, and closed (1 or 0), n each. The shape is checked first.
emscripten::val backtestTrades(std::uintptr_t eng, std::size_t T, std::size_t ndim, std::size_t ncols) {
    screamer::check_trade_rows(ndim, ncols);
    const auto trades = screamer::backtest_trades(reinterpret_cast<const double*>(eng), T,
                                                  static_cast<std::ptrdiff_t>(ncols), 1);
    const std::size_t n = trades.size();
    std::vector<double> cols(7 * n);
    for (std::size_t i = 0; i < n; ++i) {
        const auto& tr = trades[i];
        cols[i] = static_cast<double>(tr.entry);
        cols[n + i] = static_cast<double>(tr.exit);
        cols[2 * n + i] = tr.side;
        cols[3 * n + i] = static_cast<double>(tr.exit - tr.entry);
        cols[4 * n + i] = tr.pnl;
        cols[5 * n + i] = tr.cost;
        cols[6 * n + i] = tr.closed ? 1.0 : 0.0;
    }
    return emscripten::val(emscripten::typed_memory_view(cols.size(), cols.data()))
        .call<emscripten::val>("slice");
}

}  // namespace

EMSCRIPTEN_BINDINGS(screamer_backtest) {
    emscripten::function("checkEngineRows", &checkEngineRows);
    emscripten::function("backtestReportInto", &backtestReportInto);
    emscripten::function("backtestGroupsInto", &backtestGroupsInto);
    emscripten::function("backtestReportNames", &backtestReportNames);
    emscripten::function("backtestTrades", &backtestTrades);
}
