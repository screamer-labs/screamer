"""Generate the block diagrams for the multi-stream operators.

Each operator is drawn as dashed input/output "stream" boxes of square marbles
with a simple operator node in the middle. Two input streams get two colors, and
the output box shows what the operator produces (Merge interleaves the colors,
CombineLatest pairs them as split squares, Filter keeps a masked subset, Dropna
drops the NaN slots).

Run it to (re)write the SVGs used by the multi-stream notebook:

    python devtools/stream_diagrams.py

Output goes to docs/_static/diagrams/<operator>.svg.
"""
import os
import matplotlib
matplotlib.use("Agg")
# Deterministic SVG output: a fixed salt gives stable element ids, so re-running
# the generator (in `make docs` or on Read the Docs) does not churn the files.
matplotlib.rcParams["svg.hashsalt"] = "screamer-stream-diagrams"
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "docs", "_static", "diagrams")

DARK = "#0074a2"   # stream a, node label, arrows
LIGHT = "#38cced"  # stream b
PANEL = "#d7eff6"  # operator node background
DASH = "#9fc7d8"   # dashed box border and empty marble outline

S = 0.44           # marble side
GAP = 0.13         # gap between marbles
BOX_H = 1.5        # stream box height


def _marble(ax, x, y, item):
    """Draw one marble. item is {"color"}, {"pair": (c1, c2)}, or {"empty": True}."""
    if item.get("empty"):
        ax.add_patch(FancyBboxPatch((x, y), S, S, boxstyle="round,pad=0,rounding_size=0.07",
                                    facecolor="white", edgecolor=DASH, lw=1.4, zorder=3))
    elif item.get("pair"):
        c1, c2 = item["pair"]
        ax.add_patch(FancyBboxPatch((x, y), S, S, boxstyle="round,pad=0,rounding_size=0.07",
                                    facecolor=c2, edgecolor="none", zorder=3))
        ax.add_patch(Rectangle((x, y), S / 2, S, facecolor=c1, edgecolor="none", zorder=4))
    else:
        ax.add_patch(FancyBboxPatch((x, y), S, S, boxstyle="round,pad=0,rounding_size=0.07",
                                    facecolor=item["color"], edgecolor="none", zorder=3))


def stream_box(ax, cx, cy, w, title, items):
    """A dashed rounded box with a title and a centered row of marbles."""
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - BOX_H / 2), w, BOX_H,
                                boxstyle="round,pad=0,rounding_size=0.14", facecolor="none",
                                edgecolor=DASH, linestyle=(0, (4, 3)), lw=1.7, zorder=1))
    ax.text(cx, cy + BOX_H / 2 - 0.33, title, ha="center", va="center",
            color=DARK, fontsize=14, fontweight="bold")
    n = len(items)
    total = n * S + (n - 1) * GAP
    x0 = cx - total / 2
    for i, it in enumerate(items):
        _marble(ax, x0 + i * (S + GAP), cy - 0.30, it)


def op_node(ax, cx, cy, label):
    """A simple rounded operator node, sized to its label."""
    w = max(2.3, 1.1 + 0.17 * len(label))
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - 0.65), w, 1.3,
                                boxstyle="round,pad=0,rounding_size=0.16",
                                facecolor=PANEL, edgecolor="none", zorder=2))
    ax.text(cx, cy, label, ha="center", va="center", color=DARK, fontsize=15, fontweight="bold")
    return w


def arrow(ax, p0, p1):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=15, lw=2.1,
                                 color=DARK, shrinkA=3, shrinkB=3, zorder=2))


def _canvas():
    fig, ax = plt.subplots(figsize=(10.5, 3.8))
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 6)
    ax.axis("off")
    return fig, ax


def _two_in(label, out_items, out_title, a_items, b_items, a_title="stream a", b_title="stream b"):
    fig, ax = _canvas()
    stream_box(ax, 2.3, 4.3, 3.7, a_title, a_items)
    stream_box(ax, 2.3, 1.5, 3.7, b_title, b_items)
    w = op_node(ax, 7.4, 2.9, label)
    stream_box(ax, 12.3, 2.9, 3.9, out_title, out_items)
    arrow(ax, (4.2, 4.3), (7.4 - w / 2 - 0.05, 3.2))
    arrow(ax, (4.2, 1.5), (7.4 - w / 2 - 0.05, 2.6))
    arrow(ax, (7.4 + w / 2 + 0.05, 2.9), (10.35, 2.9))
    return fig


def _one_in(label, in_items, in_title, out_items, out_title):
    fig, ax = _canvas()
    stream_box(ax, 2.3, 2.9, 3.7, in_title, in_items)
    w = op_node(ax, 7.4, 2.9, label)
    stream_box(ax, 12.3, 2.9, 3.9, out_title, out_items)
    arrow(ax, (4.2, 2.9), (7.4 - w / 2 - 0.05, 2.9))
    arrow(ax, (7.4 + w / 2 + 0.05, 2.9), (10.35, 2.9))
    return fig


dark = {"color": DARK}
light = {"color": LIGHT}
empty = {"empty": True}
pair = {"pair": (DARK, LIGHT)}


def diagrams():
    return {
        "merge": _two_in("Merge", [dark, light, light, dark, light, dark, light], "merged",
                         [dark] * 6, [light] * 6),
        "combine_latest": _two_in("CombineLatest", [pair] * 5, "aligned pairs",
                                  [dark] * 4, [light] * 4),
        "filter": _two_in("Filter", [dark] * 4, "kept",
                          [dark] * 6, [dark, empty, dark, empty, dark, dark],
                          a_title="data", b_title="mask"),
        "dropna": _one_in("Dropna", [dark, empty, dark, empty, dark], "data",
                          [dark] * 3, "cleaned"),
    }


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    for name, fig in diagrams().items():
        path = os.path.join(OUT_DIR, name + ".svg")
        fig.savefig(path, bbox_inches="tight", transparent=True, metadata={"Date": None})
        plt.close(fig)
        print("wrote", os.path.normpath(path))


if __name__ == "__main__":
    main()
