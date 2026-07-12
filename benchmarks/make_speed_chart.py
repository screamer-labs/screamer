"""Build a clear speedup chart and table from the benchmark results.

For each function, take the largest series length, the fastest time per library,
and compute how many times faster screamer is than the fastest alternative
(numpy / pandas / scipy). Writes:

  - benchmarks/plots/speed_chart.png : a horizontal bar chart on a log axis,
    bars coloured by which library was the runner-up.
  - benchmarks/plots/speed_table.md  : a markdown table (vs numpy, vs pandas).

Copy the PNG into docs/img/ for the docs.
"""
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from benchmarks import read_expiriments

HERE = os.path.dirname(os.path.abspath(__file__))
COLORS = {"numpy": "#8ecae6", "pandas": "#3a7ca5", "scipy": "#adb5bd"}


# A one-million-element array: large enough that the sub-millisecond elementwise
# ops are timed cleanly. pandas is the reference for the rolling statistics (it is
# the practical tool at this scale; numpy's sliding-window is O(n*window) and only
# feasible for tiny windows here), and numpy/scipy for the pure elementwise ops.
N = 1_000_000


def speedups():
    df = read_expiriments()
    df = df[df["n"] == N]
    per_lib = df.groupby(["func", "lib"], as_index=False)["time"].min()   # best over windows
    wide = per_lib.pivot(index="func", columns="lib", values="time")
    for a in ("numpy", "pandas", "scipy"):
        if a not in wide.columns:
            wide[a] = np.nan

    def reference(r):
        if not pd.isna(r["pandas"]):
            return r["pandas"], "pandas"
        cand = {k: r[k] for k in ("numpy", "scipy") if not pd.isna(r[k])}
        best = min(cand, key=cand.get)
        return cand[best], best

    wide["best_time"], wide["best_alt"] = zip(*wide.apply(reference, axis=1))
    wide["speedup"] = wide["best_time"] / wide["screamer"]
    for a in ("numpy", "pandas"):
        wide[f"x_{a}"] = wide[a] / wide["screamer"]
    return wide.sort_values("speedup")


def chart(wide, path):
    fig, ax = plt.subplots(figsize=(9, 0.34 * len(wide) + 1.2))
    y = np.arange(len(wide))
    ax.barh(y, wide["speedup"], color=[COLORS.get(b, "gray") for b in wide["best_alt"]], zorder=3)
    ax.set_xscale("log")
    ax.axvline(1, color="k", lw=1, ls="--", alpha=0.6, zorder=2)
    ax.set_yticks(y)
    ax.set_yticklabels(wide.index)
    ax.set_ylim(-0.6, len(wide) - 0.4)
    for yi, val in zip(y, wide["speedup"]):
        ax.text(val * 1.06, yi, f"{val:.1f}x" if val < 10 else f"{val:.0f}x",
                va="center", fontsize=8, color="#333")
    ax.set_xlabel("screamer speedup vs the fastest of numpy / pandas / scipy  (log scale)")
    ax.set_title("Batch speed: screamer vs the fastest alternative", fontweight="bold")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in COLORS.values()]
    ax.legend(handles, [f"runner-up: {k}" for k in COLORS], loc="lower right", frameon=False, fontsize=9)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="x", alpha=0.25, zorder=0)
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()


def table(wide, path):
    rows = wide.sort_values("speedup", ascending=False)
    lines = ["| function | vs numpy | vs pandas | vs fastest |", "|---|---|---|---|"]
    def cell(v):
        return "-" if pd.isna(v) else (f"{v:.1f}x" if v < 10 else f"{v:.0f}x")
    for func, r in rows.iterrows():
        lines.append(f"| `{func}` | {cell(r.get('x_numpy', np.nan))} | "
                     f"{cell(r.get('x_pandas', np.nan))} | **{cell(r['speedup'])}** |")
    open(path, "w").write("\n".join(lines) + "\n")


def main():
    wide = speedups()
    os.makedirs(os.path.join(HERE, "plots"), exist_ok=True)
    chart(wide, os.path.join(HERE, "plots", "speed_chart.png"))
    table(wide, os.path.join(HERE, "plots", "speed_table.md"))
    faster = (wide["speedup"] >= 1.0).mean()
    print(f"{len(wide)} functions; screamer fastest on {faster:.0%}; "
          f"median speedup {wide['speedup'].median():.1f}x, max {wide['speedup'].max():.0f}x")
    print("wrote plots/speed_chart.png and plots/speed_table.md")


if __name__ == "__main__":
    main()
