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

constexpr const char* kWho = "backtestReport";

// The shape rule of backtestReport's input ((T, 4), or (T, K, 4) with
// K >= 1; `d1` and `d2` are the second and third sizes, 0 where absent),
// checked first, so a bad call raises the same error class as in Python
// whatever else is wrong with it.
void checkReportRows(std::size_t ndim, std::size_t d1, std::size_t d2) {
    screamer::check_report_rows(kWho, ndim, d1, d2);
}

// Copies T row-major rows of `width` values to `o` as columns of T values;
// returns the end of the written columns.
double* toColumns(const double* rows, std::size_t T, std::size_t width, double* o) {
    for (std::size_t c = 0; c < width; ++c, o += T)
        for (std::size_t t = 0; t < T; ++t) o[t] = rows[t * width + c];
    return o;
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
    double* o = toColumns(report.data(), T, kReport, reinterpret_cast<double*>(out));
    toColumns(stats.data(), T, kStatsCols, o);
}

// A portfolio engine's output `eng` (T x K rows of 4 values, row-major) as
// the book's running columns, column-major in `out` (3 + 6 + 9 columns of T
// values: kBookColumns, then the order of backtestReportNames), the book's
// summary in `summary` (kNumStats values), and each asset's summary in
// `per_asset` (K x kNumStats, row-major). The shape and periods_per_year are
// checked first.
void backtestBookInto(std::uintptr_t eng, std::size_t T, std::size_t K, double periods_per_year,
                      std::uintptr_t out, std::uintptr_t summary, std::uintptr_t per_asset) {
    screamer::check_report_rows(kWho, 3, K, kEngine);
    screamer::check_periods_per_year(periods_per_year);
    constexpr std::size_t kBook = screamer::kBookColumns.size();
    std::vector<double> book(T * kBook), report(T * kReport), stats(T * kStatsCols);
    screamer::backtest_book_report(reinterpret_cast<const double*>(eng), T, K,
                                   static_cast<std::ptrdiff_t>(K * kEngine),
                                   static_cast<std::ptrdiff_t>(kEngine), 1, periods_per_year,
                                   book.data(), report.data(), stats.data(),
                                   reinterpret_cast<double*>(summary),
                                   reinterpret_cast<double*>(per_asset));
    double* o = toColumns(book.data(), T, kBook, reinterpret_cast<double*>(out));
    o = toColumns(report.data(), T, kReport, o);
    toColumns(stats.data(), T, kStatsCols, o);
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

// backtestGroupsInto for the book of a portfolio engine's output `eng` (T x K
// rows of 4 values, row-major).
void backtestBookGroupsInto(std::uintptr_t eng, std::size_t T, std::size_t K, std::uintptr_t codes,
                            std::size_t n_codes, std::size_t groups, double periods_per_year,
                            std::uintptr_t out) {
    screamer::check_report_rows(kWho, 3, K, kEngine);
    screamer::check_group_codes(n_codes, T);
    const double* c = reinterpret_cast<const double*>(codes);
    std::vector<std::int64_t> ic(T);
    for (std::size_t t = 0; t < T; ++t) ic[t] = static_cast<std::int64_t>(c[t]);
    screamer::backtest_book_groups(reinterpret_cast<const double*>(eng), T, K,
                                   static_cast<std::ptrdiff_t>(K * kEngine),
                                   static_cast<std::ptrdiff_t>(kEngine), 1, ic.data(), groups,
                                   periods_per_year, reinterpret_cast<double*>(out));
}

// [engine columns, BacktestReport columns, BacktestStats columns, summary,
// group statistics, book columns].
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
    all.set(5, names(screamer::kBookColumns));
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
    emscripten::function("checkReportRows", &checkReportRows);
    emscripten::function("backtestReportInto", &backtestReportInto);
    emscripten::function("backtestBookInto", &backtestBookInto);
    emscripten::function("backtestGroupsInto", &backtestGroupsInto);
    emscripten::function("backtestBookGroupsInto", &backtestBookGroupsInto);
    emscripten::function("backtestReportNames", &backtestReportNames);
    emscripten::function("backtestTrades", &backtestTrades);
}
