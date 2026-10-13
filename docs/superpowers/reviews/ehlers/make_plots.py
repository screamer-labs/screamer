"""Regenerate the plots of the F3 Ehlers review pack.

Run from the repository root:

    .venv/bin/python docs/superpowers/reviews/ehlers/make_plots.py

The signals and helpers come from tests/test_ehlers_known_signals.py, so the
plots show exactly what the automated checks measure. TA-Lib is drawn as a
reference where it implements the same Rocket Science routine.
"""

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
import numpy as np  # noqa: E402
import talib  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
sys.path.insert(0, ROOT)

import screamer as s  # noqa: E402
from tests.test_ehlers_known_signals import (  # noqa: E402
    PERIODS, SETTLE, chirp, gain_db, sine, sine_plus_trend, unwrap_deg,
)

OUT = os.path.dirname(os.path.abspath(__file__))
C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"   # categorical slots 1-3
EXPECT = "#0b0b0b"                              # expected value: dark dashed
BAND = "#d9d8d4"                                # tolerance band
MUTED = "#52514e"

plt.rcParams.update({
    "figure.dpi": 100, "font.size": 9, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": "#ecebe8",
    "grid.linewidth": 0.6, "lines.linewidth": 1.4, "legend.frameon": False,
})


def save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, name), dpi=90)
    plt.close(fig)


def expect_line(ax, y, label, band=None):
    if band is not None:
        ax.axhspan(y * (1 - band), y * (1 + band), color=BAND, alpha=0.6, lw=0)
    ax.axhline(y, color=EXPECT, ls="--", lw=1)
    ax.annotate(label, (1.0, y), xycoords=("axes fraction", "data"),
                xytext=(-4, 3), textcoords="offset points", ha="right",
                color=EXPECT, fontsize=8)


def settle_marker(ax):
    ax.axvline(SETTLE, color=MUTED, lw=0.8, ls=":")


# ------------------------------------------------ DominantCycle, sine
fig, ax = plt.subplots(figsize=(7, 3.4))
for P, c in zip(PERIODS, (C1, C2, C3)):
    ax.plot(s.DominantCycle()(sine(P)), color=c, label=f"P = {P:g}")
    expect_line(ax, P, f"expected {P:g} (5% band)", band=0.05)
settle_marker(ax)
ax.set(title="DominantCycle on a pure sine", xlabel="sample", ylabel="period",
       xlim=(0, 600), ylim=(0, 50))
ax.legend(loc="lower right", ncol=3)
save(fig, "dominant_cycle_sine.png")

# ------------------------------------------------ DominantCycle, chirp
x, P = chirp()
fig, ax = plt.subplots(figsize=(7, 3.4))
ax.fill_between(np.arange(len(P)), P * 0.9, P * 1.1, color=BAND, alpha=0.6, lw=0,
                label="true period, 10% band")
ax.plot(P, color=EXPECT, ls="--", lw=1, label="true period")
ax.plot(s.DominantCycle()(x), color=C1, label="DominantCycle")
ax.set(title="DominantCycle on a chirp, period 15 to 40", xlabel="sample",
       ylabel="period", ylim=(0, 50))
ax.legend(loc="lower right")
save(fig, "dominant_cycle_chirp.png")

# ------------------------------------------- DominantCycle, sine + trend
x, _ = sine_plus_trend(10.0)
fig, ax = plt.subplots(figsize=(7, 3.4))
ax.plot(talib.HT_DCPERIOD(x), color=MUTED, lw=3, alpha=0.4, label="TA-Lib HT_DCPERIOD, raw")
ax.plot(s.DominantCycle()(x), color=C2, label="DominantCycle, raw input")
roof = s.RoofingFilter(hp_period=48.0, lp_period=8.0)(x)
ax.plot(s.DominantCycle()(roof), color=C1, label="DominantCycle after RoofingFilter(48, 8)")
expect_line(ax, 10.0, "expected 10 (5% band)", band=0.05)
settle_marker(ax)
ax.set(title="DominantCycle on sine (P = 10) plus a ramp of one amplitude per cycle",
       xlabel="sample", ylabel="period", xlim=(0, 600), ylim=(0, 20))
ax.legend(loc="lower right")
save(fig, "dominant_cycle_trend.png")

# ------------------------------------------------ CycleFrequency
fig, ax = plt.subplots(figsize=(7, 3.4))
for P, c in zip(PERIODS, (C1, C2, C3)):
    ax.plot(s.CycleFrequency()(sine(P)), color=c, label=f"P = {P:g}")
    expect_line(ax, 1.0 / P, f"expected 1/{P:g}", band=0.05)
settle_marker(ax)
ax.set(title="CycleFrequency on a pure sine", xlabel="sample",
       ylabel="cycles per sample", xlim=(0, 600), ylim=(0, 0.2))
ax.legend(loc="upper right", ncol=3)
save(fig, "cycle_frequency_sine.png")

# ------------------------------------------------ CycleAmplitude
fig, ax = plt.subplots(figsize=(7, 3.4))
for P, c in zip(PERIODS, (C1, C2, C3)):
    a = s.CycleAmplitude()(sine(P))
    ax.plot(a, color=c, label=f"P = {P:g} (median {np.median(a[SETTLE:]):.2f})")
expect_line(ax, 1.0, "true amplitude 1 (5% band)", band=0.05)
settle_marker(ax)
ax.set(title="CycleAmplitude on a unit sine (FAIL)", xlabel="sample",
       ylabel="amplitude", xlim=(0, 600), ylim=(0, 1.8))
ax.legend(loc="center left", bbox_to_anchor=(0.35, 0.42))
save(fig, "cycle_amplitude_sine.png")

# ------------------------------------------------ HilbertPhasor
P = 20.0
out = s.HilbertPhasor()(sine(P))
fig, ax = plt.subplots(figsize=(7, 3.4))
t = np.arange(len(out))
w = slice(400, 460)
ax.plot(t[w], out[w, 0], color=C1, marker="o", ms=3, label="inphase I")
ax.plot(t[w], out[w, 1], color=C2, marker="o", ms=3, label="quadrature Q")
pk = 400 + int(np.argmax(out[400:420, 0]))
ax.axvline(pk, color=C1, lw=0.8, ls=":")
ax.axvline(pk - P / 4, color=EXPECT, lw=1, ls="--")
ax.annotate("expected Q peak: I peak - P/4", (pk - P / 4, 1.15), xytext=(4, 0),
            textcoords="offset points", fontsize=8, color=EXPECT)
ax.set(title="HilbertPhasor on a unit sine, P = 20 (Q leads I by a quarter cycle)",
       xlabel="sample", ylabel="component", ylim=(-1.4, 1.4))
ax.legend(loc="lower right")
save(fig, "hilbert_phasor.png")

# ------------------------------------------------ CyclePhase
fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.2))
P = 20.0
ph = s.CyclePhase()(sine(P))
w = slice(400, 460)
axes[0].plot(t[w], ph[w], color=C1, marker="o", ms=3, label="CyclePhase")
axes[0].set(title="phase, P = 20", xlabel="sample", ylabel="degrees", ylim=(0, 360))
for P_, c in zip(PERIODS, (C1, C2, C3)):
    d = unwrap_deg(np.diff(s.CyclePhase()(sine(P_))))
    axes[1].plot(d, color=c, label=f"P = {P_:g}")
    expect_line(axes[1], -360.0 / P_, f"-360/{P_:g}", band=0.05)
axes[1].set(title="phase change per sample", xlabel="sample", xlim=(0, 600),
            ylim=(-60, 10))
axes[1].legend(loc="lower right", ncol=3, fontsize=7)
save(fig, "cycle_phase.png")

# ------------------------------------------------ CycleSine
P = 20.0
x = sine(P)
cs = s.CycleSine()(x)
ts, tl = talib.HT_SINE(x)
fig, axes = plt.subplots(2, 1, figsize=(7, 4.6), sharex=True)
w = slice(400, 441)
axes[0].plot(t[w], cs[w, 0], color=C1, label="sine")
axes[0].plot(t[w], cs[w, 1], color=C2, label="leadsine")
axes[0].set(title="screamer CycleSine, P = 20 (FAIL: leadsine lags)", ylim=(-1.3, 1.3))
axes[1].plot(t[w], ts[w], color=C1, label="sine")
axes[1].plot(t[w], tl[w], color=C2, label="leadsine")
axes[1].set(title="TA-Lib HT_SINE, same input (leadsine leads)", xlabel="sample",
            ylim=(-1.3, 1.3))
for ax, sn in zip(axes, (cs[:, 0], ts)):
    pk = 400 + int(np.argmax(sn[400:420]))
    ax.axvline(pk, color=C1, lw=0.8, ls=":")
    ax.axvline(pk - P / 8, color=EXPECT, lw=1, ls="--")
    ax.annotate("expected leadsine peak: sine peak - P/8", (pk - P / 8, -1.2),
                xytext=(4, 0), textcoords="offset points", ha="left", fontsize=8,
                color=EXPECT)
    ax.legend(loc="upper right", ncol=2)
save(fig, "cycle_sine.png")

# ------------------------------------------------ TrendMode
fig, axes = plt.subplots(2, 1, figsize=(7, 4.6), sharex=True)
x, _ = sine_plus_trend(20.0)
tm = s.TrendMode()(x)
axes[0].step(t[: len(tm)], tm, color=C1, where="post", label="TrendMode")
axes[0].axhline(0.0, color=EXPECT, ls="--", lw=1)
axes[0].set(title=f"sine (P = 20) plus a ramp: expected 0 (cycle); "
                  f"cycle on {np.mean(tm[SETTLE:] == 0):.0%}",
            ylim=(-0.2, 1.2))
ramp = 0.05 * np.arange(len(t), dtype=float)
tm = s.TrendMode()(ramp)
axes[1].step(t, tm, color=C2, where="post", label="TrendMode")
axes[1].step(t, talib.HT_TRENDMODE(ramp), color=MUTED, where="post", alpha=0.5,
             label="TA-Lib HT_TRENDMODE")
axes[1].axhline(1.0, color=EXPECT, ls="--", lw=1)
axes[1].set(title=f"pure ramp: expected 1 (trend); trend on "
                  f"{np.mean(tm[SETTLE:] == 1):.0%} (FAIL)",
            xlabel="sample", xlim=(0, 400), ylim=(-0.2, 1.2))
axes[1].legend(loc="center right", frameon=True, facecolor="white", edgecolor="white")
save(fig, "trend_mode.png")

# ------------------------------------------------ InstantaneousTrendline
fig, ax = plt.subplots(figsize=(7, 3.4))
x, trend = sine_plus_trend(20.0)
ax.plot(x, color=BAND, lw=2, label="input")
ax.plot(trend, color=EXPECT, ls="--", lw=1, label="true trend (expected)")
ax.plot(s.InstantaneousTrendline()(x), color=C1, label="InstantaneousTrendline")
ax.plot(talib.HT_TRENDLINE(x), color=C2, lw=1, label="TA-Lib HT_TRENDLINE")
ax.set(title="InstantaneousTrendline on sine (P = 20) plus a ramp (FAIL)",
       xlabel="sample", xlim=(300, 420), ylim=(13, 22))
ax.legend(loc="upper left")
save(fig, "instantaneous_trendline.png")

# ------------------------------------------------ MAMA
fig, ax = plt.subplots(figsize=(7, 3.4))
for P_, c in zip(PERIODS, (C1, C2, C3)):
    x = sine(P_) + 5.0
    m = s.MAMA()(x)[:, 0]
    alpha = np.diff(m) / (x[1:] - m[:-1])
    ax.plot(np.arange(1, len(m)), alpha, color=c, lw=1, label=f"P = {P_:g}")
    e = float(np.clip(0.5 / (360.0 / P_), 0.05, 0.5))
    ax.axhline(e, color=EXPECT, ls="--", lw=1)
ax.annotate("expected clip(0.5 P / 360, 0.05, 0.5): 0.05, 0.05, 0.056", (1.0, 0.07),
            xycoords=("axes fraction", "data"), ha="right", fontsize=8, color=EXPECT)
ax.set(title="MAMA effective alpha on a sine (spikes: one phase wrap per cycle)",
       xlabel="sample", ylabel="alpha", xlim=(300, 420), yscale="log", ylim=(0.03, 0.7))
ax.set_yticks([0.05, 0.1, 0.2, 0.5])
ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
ax.legend(loc="center right")
save(fig, "mama.png")

# ------------------------------------------------ filters
periods = np.geomspace(4, 200, 60)
fig, axes = plt.subplots(1, 3, figsize=(9, 3.2), sharey=True)
specs = [
    ("SuperSmoother(period=20)", lambda: s.SuperSmoother(period=20.0), [20.0]),
    ("Decycler(period=60)", lambda: s.Decycler(period=60.0), [60.0]),
    ("RoofingFilter(48, 10)", lambda: s.RoofingFilter(hp_period=48.0, lp_period=10.0),
     [48.0, 10.0]),
]
for ax, (name, make, cuts) in zip(axes, specs):
    g = [gain_db(make(), p, n=2000) for p in periods]
    ax.plot(periods, g, color=C1)
    ax.axhspan(-4, -2, color=BAND, alpha=0.6, lw=0)
    ax.axhline(-3, color=EXPECT, ls="--", lw=1)
    for cpt in cuts:
        ax.axvline(cpt, color=MUTED, lw=0.8, ls=":")
        ax.plot([cpt], [gain_db(make(), cpt)], "o", color=C2, ms=6)
    ax.set(title=name, xscale="log", xlabel="period (samples)", ylim=(-30, 3))
axes[0].set_ylabel("gain (dB)")
axes[0].annotate("-3 dB +/- 1 dB", (5, -2.4), fontsize=8, color=EXPECT)
save(fig, "filters_gain.png")

print("done")
