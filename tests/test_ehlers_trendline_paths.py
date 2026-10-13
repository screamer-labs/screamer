"""The two forms of detail::CycleTrendline give the same bits.

TrendMode sums its trendline window with update() and InstantaneousTrendline
with prepare() and finish(); each form is the faster one inside its operator.
Both must equal TA-Lib HT_TRENDLINE's newest-first sum, so they must also equal
each other bit for bit. Neither form is reachable from Python on its own, so
this test compiles a small C++ driver against the headers (as
test_pure_headers does) and feeds both forms the same period from one
HilbertCycle engine over random and hostile inputs.
"""

import pathlib
import shutil
import subprocess

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent

DRIVER = r"""
#include "screamer/detail/hilbert_cycle.h"
#include "screamer/detail/ehlers_sinewave.h"
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

using screamer::detail::CycleTrendline;
using screamer::detail::HilbertCycle;

static std::uint64_t bits(double v) {
    std::uint64_t b;
    std::memcpy(&b, &v, sizeof b);
    return b;
}

// Feeds one series to both forms; NaN inputs are skipped, as the operators
// skip them. Returns the number of samples whose values differ in any bit.
static long compare(const std::vector<double>& x) {
    HilbertCycle engine;
    CycleTrendline a, b;
    long bad = 0;
    for (double v : x) {
        if (std::isnan(v)) continue;
        b.prepare(v);
        engine.update(v);
        const double sp = engine.raw_smooth_period();
        a.update(v, sp);
        b.finish(sp);
        if (bits(a.value()) != bits(b.value())) ++bad;
    }
    return bad;
}

int main() {
    std::mt19937_64 g(7);
    std::normal_distribution<double> n01;
    std::uniform_real_distribution<double> u01(0.0, 1.0);
    std::vector<std::vector<double>> series;
    const int len = 20000;

    // Random walks at several levels and step sizes.
    for (double level : {0.0, 100.0, 1e6}) {
        for (double step : {1e-3, 1.0, 1e3}) {
            std::vector<double> x(len);
            double c = level;
            for (double& v : x) { c += step * n01(g); v = c; }
            series.push_back(x);
        }
    }
    // A walk whose step scale switches between 1e-3 and 1e3, with spikes.
    {
        std::vector<double> x(len);
        double c = 100.0, scale = 1.0;
        for (int i = 0; i < len; ++i) {
            if (i % 500 == 0) scale = (scale == 1e-3) ? 1e3 : 1e-3;
            c += scale * n01(g);
            x[i] = (u01(g) < 0.002) ? c + (u01(g) < 0.5 ? -1e6 : 1e6) : c;
        }
        series.push_back(x);
    }
    // Constant, zero, alternating sign, tiny and huge magnitudes.
    series.push_back(std::vector<double>(2000, 42.0));
    series.push_back(std::vector<double>(2000, 0.0));
    {
        std::vector<double> x(2000);
        for (int i = 0; i < 2000; ++i) x[i] = (i % 2) ? 1.0 : -1.0;
        series.push_back(x);
    }
    {
        std::vector<double> x(2000), y(2000);
        for (int i = 0; i < 2000; ++i) {
            x[i] = 1e-300 * (1.0 + u01(g));
            y[i] = 1e300 * (1.0 + u01(g));
        }
        series.push_back(x);
        series.push_back(y);
    }
    // NaN gaps, and a stretch of infinities.
    {
        std::vector<double> x(4000);
        double c = 50.0;
        for (int i = 0; i < 4000; ++i) {
            c += n01(g);
            x[i] = (i % 37 == 0) ? NAN : c;
        }
        series.push_back(x);
        for (int i = 1000; i < 1010; ++i) x[i] = (i % 2) ? INFINITY : -INFINITY;
        series.push_back(x);
    }

    long total = 0, samples = 0;
    for (const auto& x : series) {
        total += compare(x);
        samples += static_cast<long>(x.size());
    }
    std::printf("%ld %ld\n", total, samples);
    return 0;
}
"""


def _compiler():
    for cxx in ("c++", "clang++", "g++"):
        path = shutil.which(cxx)
        if path:
            return path
    return None


def test_update_and_prepare_finish_are_bit_identical(tmp_path):
    cxx = _compiler()
    if cxx is None:
        pytest.skip("no C++ compiler on PATH")
    src = tmp_path / "trendline_paths.cpp"
    exe = tmp_path / "trendline_paths"
    src.write_text(DRIVER)
    build = subprocess.run(
        [cxx, "-std=c++17", "-O2", "-ffp-contract=off", "-I", str(ROOT / "include"),
         str(src), "-o", str(exe)],
        capture_output=True, text=True)
    assert build.returncode == 0, build.stderr
    run = subprocess.run([str(exe)], capture_output=True, text=True)
    assert run.returncode == 0, run.stderr
    bad, samples = (int(v) for v in run.stdout.split())
    assert samples > 200000
    assert bad == 0, f"{bad} of {samples} samples differ between the two forms"
