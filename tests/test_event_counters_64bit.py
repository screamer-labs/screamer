"""Event counters do not wrap after 2**31 events of a live stream.

A live stream at 1000 events per second passes 2**31 events in 25 days. Each
operator below counts its events (a warm-up counter or a sliding-window
sample index); a 32-bit counter wraps there, which sends a warm-up test back
to "not ready" (NaN forever) or expires the wrong window entries.

Running 2**31 events per operator takes too long for the suite, so a C++
harness warms one copy of each operator up, makes a second copy whose counters
are 2**31 events further on (the state that many more events leave, set
through the test-only friend detail::TestAccess), and feeds both the same
events: past the warm-up, the count changes nothing, so the two must agree bit
for bit. It compiles C++ from the source tree, so the wheel test job skips it
(build-wheels.yml), as it skips the other compile gates.
"""
import pathlib
import shutil
import subprocess

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]

DRIVER = r"""
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
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

using namespace screamer;
static const std::int64_t K = std::int64_t(1) << 31;
static int failures = 0;

// The classes befriend detail::TestAccess (screamer/detail/test_access.h):
// it moves each event counter on by 2**31, the state that many more events
// leave.
namespace screamer { namespace detail {
struct TestAccess {
    template <bool M>
    static void bump(MonotonicDeque<M>& d) {
        d.index_ += K;
        for (auto& i : d.indices_) i += K;
    }
    static void bump(KAMA& o) { o.n_seen_ += K; }
    static void bump(RollingYangZhangVar& o) { o.n_seen_ += K; }
    static void bump(Stoch& o) { o.n_seen_ += K; bump(o.max_deque_); bump(o.min_deque_); }
    static void bump(StochRSI& o) { o.n_valid_rsi_ += K; bump(o.max_deque_); bump(o.min_deque_); }
    static void bump(MFI& o) { o.n_directional_ += K; }
    static void bump(ATR& o) { o.n_tr_ += K; }
    static void bump(RollingRSI& o) { o.wilder_count_ += K; }
    static void bump(RollingMax& o) { bump(o.deque_); }
    static void bump(RollingArgmin& o) { bump(o.deque_); }
    static void bump(WilliamsR& o) { bump(o.max_deque_); bump(o.min_deque_); }
    static void bump(ADX& o) { o.core_.n_seen_ += K; }
    static void bump(MAMA& o) { o.engine_.n_ += K; }
    static void bump(WilderSmoother& o) { o.n_seen_ += K; }
    // ExpandingMean: 2**31 more events of the same mean.
    static void bump(ExpandingMean& o) {
        o.sum_x_ *= double(K + o.n_) / double(o.n_);
        o.n_ += K;
    }
};
}}  // namespace screamer::detail
using detail::TestAccess;

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
// counters on by 2**31, and checks both on the second half. `cols` picks the
// operator's inputs from a bar.
template <class Op>
void check(const char* name, Op a, std::vector<int> cols) {
    const auto bs = bars(400);
    std::vector<double> in(cols.size()), oa(a.n_out()), ob(a.n_out());
    for (std::size_t i = 0; i < 200; ++i) {
        for (std::size_t c = 0; c < cols.size(); ++c) in[c] = bs[i][cols[c]];
        a.eval(in.data(), oa.data());
    }
    Op b = a;
    TestAccess::bump(b);
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
    const std::vector<int> C{3}, HLC{1, 2, 3}, OHLC{0, 1, 2, 3}, HLCV{1, 2, 3, 4};
    check("KAMA", KAMA(10), C);
    check("RollingYangZhangVar", RollingYangZhangVar(10), OHLC);
    check("Stoch", Stoch(14, 3, 3), HLC);
    check("StochRSI", StochRSI(14), C);
    check("MFI", MFI(14), HLCV);
    check("ATR", ATR(14), HLC);
    check("RollingRSI", RollingRSI(14), C);
    check("RollingMax", RollingMax(5), C);
    check("RollingArgmin", RollingArgmin(5), C);
    check("WilliamsR", WilliamsR(14), HLC);
    check("ADX", ADX(14), HLC);
    check("MAMA", MAMA(), C);
    {
        ExpandingMean a;
        for (int i = 0; i < 10; ++i) a.process_scalar(2.0);
        TestAccess::bump(a);
        const double m = a.process_scalar(2.0);
        if (m != 2.0) { std::printf("ExpandingMean: %.17g\n", m); ++failures; }
    }
    {
        detail::WilderSmoother a(14);
        for (int i = 0; i < 50; ++i) a.append(1.0 + i % 3);
        detail::WilderSmoother b = a;
        TestAccess::bump(b);
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
