"""Event counters do not wrap after 2**31 events of a live stream.

A live stream at 1000 events per second passes 2**31 events in 25 days. Each
operator below counts its events (a warm-up counter or a sliding-window
sample index); a 32-bit counter wraps there, which sends a warm-up test back
to "not ready" (NaN forever) or expires the wrong window entries.

Running 2**31 events per operator takes too long for the suite, so a C++
harness warms one copy of each operator up, makes a second copy whose counters
are 2**31 events further on (the state that many more events leave), and feeds
both the same events: past the warm-up, the count changes nothing, so the two
must agree bit for bit.
"""
import pathlib
import shutil
import subprocess

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]

DRIVER = r"""
// The counters are private: the harness reads them as public members.
#define private public
#define protected public
#include "screamer/kama.h"
#include "screamer/yang_zhang.h"
#include "screamer/stoch.h"
#include "screamer/stoch_rsi.h"
#include "screamer/mfi.h"
#include "screamer/atr.h"
#include "screamer/rolling_rsi.h"
#include "screamer/rolling_max.h"
#include "screamer/rolling_argmin.h"
#include "screamer/williams_r.h"
#include "screamer/adx.h"
#include "screamer/mama.h"
#include "screamer/expanding_mean.h"
#include "screamer/detail/wilder_smoother.h"
#undef private
#undef protected
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

using namespace screamer;
static const std::int64_t K = std::int64_t(1) << 31;
static int failures = 0;

template <bool M>
void bump(detail::MonotonicDeque<M>& d) {
    d.index_ += K;
    for (auto& i : d.indices_) i += K;
}

// Bars: open, high, low, close, volume.
static std::vector<std::vector<double>> bars(std::size_t n) {
    std::mt19937_64 g(7);
    std::normal_distribution<double> z(0.0, 1.0);
    std::vector<std::vector<double>> out;
    double c = 100.0;
    for (std::size_t i = 0; i < n; ++i) {
        const double o = c, nc = c + z(g);
        const double h = std::max(o, nc) + std::fabs(z(g)), l = std::min(o, nc) - std::fabs(z(g));
        out.push_back({o, h, l, nc, 1.0 + std::fabs(z(g))});
        c = nc;
    }
    return out;
}

// Warms `a` up on the first half of the bars, copies it into `b`, moves b's
// counters on by 2**31 with `bump`, and checks both on the second half.
// `cols` picks the operator's inputs from a bar.
template <class Op, class Bump>
void check(const char* name, Op a, std::vector<int> cols, Bump bump_fn) {
    const auto bs = bars(400);
    std::vector<double> in(cols.size()), oa(a.n_out()), ob(a.n_out());
    for (std::size_t i = 0; i < 200; ++i) {
        for (std::size_t c = 0; c < cols.size(); ++c) in[c] = bs[i][cols[c]];
        a.eval(in.data(), oa.data());
    }
    Op b = a;
    bump_fn(b);
    for (std::size_t i = 200; i < bs.size(); ++i) {
        for (std::size_t c = 0; c < cols.size(); ++c) in[c] = bs[i][cols[c]];
        a.eval(in.data(), oa.data());
        b.eval(in.data(), ob.data());
        if (std::memcmp(oa.data(), ob.data(), oa.size() * sizeof(double)) != 0 ||
                std::isnan(oa[0])) {
            std::printf("%s: event %zu gives %.17g after 2**31 more events, %.17g without\n",
                        name, i, ob[0], oa[0]);
            ++failures;
            return;
        }
    }
}

int main() {
    const std::vector<int> C{3}, HLC{1, 2, 3}, HL{1, 2}, OHLC{0, 1, 2, 3}, HLCV{1, 2, 3, 4};
    check("KAMA", KAMA(10), C, [](KAMA& o) { o.n_seen_ += K; });
    check("RollingYangZhangVar", RollingYangZhangVar(10), OHLC,
          [](RollingYangZhangVar& o) { o.n_seen_ += K; });
    check("Stoch", Stoch(14, 3, 3), HLC, [](Stoch& o) {
        o.n_seen_ += K; bump(o.max_deque_); bump(o.min_deque_); });
    check("StochRSI", StochRSI(14), C, [](StochRSI& o) {
        o.n_valid_rsi_ += K; bump(o.max_deque_); bump(o.min_deque_); });
    check("MFI", MFI(14), HLCV, [](MFI& o) { o.n_directional_ += K; });
    check("ATR", ATR(14), HLC, [](ATR& o) { o.n_tr_ += K; });
    check("RollingRSI", RollingRSI(14), C, [](RollingRSI& o) { o.wilder_count_ += K; });
    check("RollingMax", RollingMax(5), C, [](RollingMax& o) { bump(o.deque_); });
    check("RollingArgmin", RollingArgmin(5), C, [](RollingArgmin& o) { bump(o.deque_); });
    check("WilliamsR", WilliamsR(14), HLC, [](WilliamsR& o) {
        bump(o.max_deque_); bump(o.min_deque_); });
    check("ADX", ADX(14), HLC, [](ADX& o) { o.core_.n_seen_ += K; });
    check("MAMA", MAMA(), C, [](MAMA& o) { o.engine_.n_ += K; });

    // ExpandingMean: the count is the divisor, so 2**31 more events of the
    // same mean leave the mean where it is.
    {
        ExpandingMean a;
        for (int i = 0; i < 10; ++i) a.process_scalar(2.0);
        a.sum_x_ *= double(K + 10) / 10.0;
        a.n_ += K;
        const double m = a.process_scalar(2.0);
        if (m != 2.0) { std::printf("ExpandingMean: %.17g\n", m); ++failures; }
    }

    // WilderSmoother on its own (the shared Wilder recursion).
    {
        detail::WilderSmoother a(14);
        for (int i = 0; i < 50; ++i) a.append(1.0 + i % 3);
        detail::WilderSmoother b = a;
        b.n_seen_ += K;
        for (int i = 0; i < 50; ++i) {
            const double x = a.append(2.0 + i % 5), y = b.append(2.0 + i % 5);
            if (x != y) { std::printf("WilderSmoother: %.17g vs %.17g\n", y, x); ++failures; break; }
        }
    }
    std::printf("failures %d\n", failures);
    return failures != 0;
}
"""


def _compiler():
    for cxx in ("c++", "clang++", "g++"):
        path = shutil.which(cxx)
        if path:
            return path
    return None


def test_event_counters_do_not_wrap_after_2_31_events(tmp_path):
    cxx = _compiler()
    if cxx is None:
        pytest.skip("no C++ compiler on PATH")
    src = tmp_path / "counters.cpp"
    exe = tmp_path / "counters"
    src.write_text(DRIVER)
    common = ROOT / "src" / "screamer" / "common"
    build = subprocess.run(
        [cxx, "-std=c++17", "-O1", "-I", str(ROOT / "include"), str(src),
         str(common / "base.cpp"), str(common / "functor_base.cpp"),
         str(ROOT / "src" / "screamer" / "detail" / "start_policy.cpp"),
         "-o", str(exe)],
        capture_output=True, text=True)
    assert build.returncode == 0, build.stderr[-4000:]
    run = subprocess.run([str(exe)], capture_output=True, text=True)
    assert run.returncode == 0, run.stdout + run.stderr
