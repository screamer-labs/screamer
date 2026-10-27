#!/usr/bin/env python3
"""Build the visual function gallery: thumbnails plus docs/gallery.rst.

Every function page under ``docs/functions_*/`` that has a ``.. plotly::``
example gets a small static thumbnail. The script runs the page's first plotly
block, captures the figure it shows, redraws its traces with matplotlib, and
saves a 320 x 200 WebP image under ``docs/_static/gallery/``. It then writes
``docs/gallery.rst``, a card grid grouped by the function groups in
``docs/topics.yml``, where each thumbnail links to its page.

The redraw uses matplotlib, which the docs already depend on, so the script
needs no browser and no extra package. It covers the trace types the docs use
(scatter lines and markers, fills, bars, histograms, candlesticks, horizontal
and vertical reference lines); it is a sketch of the interactive plot, not a
pixel copy.

Thumbnails are cached by a hash of the plot code. A rebuild only re-runs the
pages whose code changed, so it is cheap. Pass ``--force`` to redraw all.

Run after adding or changing a function-page plot, and commit the result:

    poetry run python devtools/build_gallery.py
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import os
import re
import sys
import textwrap
import warnings
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from devtools.topics import load_groups  # noqa: E402

DOCS = ROOT / "docs"
OUT_DIR = DOCS / "_static" / "gallery"
MANIFEST = OUT_DIR / "manifest.json"
GALLERY_RST = DOCS / "gallery.rst"

# Bump when the renderer changes, so every cached thumbnail is redrawn.
RENDERER_VERSION = "1"
THUMB_W, THUMB_H = 320, 200

FRONTMATTER_RE = re.compile(r"\A---\n(.*?)\n---\n", re.S)
PLOTLY_RE = re.compile(
    r"^```\{eval-rst\}\n\.\. plotly::[^\n]*\n((?:[ \t]+:[^\n]*\n)*)\s*\n(.*?)^```",
    re.S | re.M,
)

# Plotly's default colorway, used when a trace sets no color.
COLORWAY = ["#636efa", "#EF553B", "#00cc96", "#ab63fa", "#FFA15A",
            "#19d3f3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52"]
DASHES = {"dash": "--", "dot": ":", "dashdot": "-.", "longdash": "--",
          "longdashdot": "-."}


# --------------------------------------------------------------------------
# Page discovery
# --------------------------------------------------------------------------

def extract_plot_code(text: str) -> str | None:
    """Return the dedented body of the first ``.. plotly::`` block, or None."""
    m = PLOTLY_RE.search(text)
    if not m:
        return None
    # The directive body is the indented block; an unindented line (prose or a
    # second directive in the same eval-rst fence) ends it.
    body = []
    for line in m.group(2).splitlines():
        if line.strip() and not line[0].isspace():
            break
        body.append(line)
    return textwrap.dedent("\n".join(body)).strip() + "\n"


def function_pages() -> list[dict]:
    """Every function page with a plotly example, with its name and topics."""
    pages = []
    for md in sorted(DOCS.glob("functions_*/*.md")):
        text = md.read_text()
        fm_match = FRONTMATTER_RE.match(text)
        fm = yaml.safe_load(fm_match.group(1)) if fm_match else {}
        fm = fm or {}
        code = extract_plot_code(text)
        if code is None:
            continue
        pages.append({
            "path": md,
            "doc": f"{md.parent.name}/{md.stem}",
            "name": fm.get("name") or md.stem,
            "topics": fm.get("topics") or [],
            "code": code,
        })
    return pages


def thumb_name(doc: str) -> str:
    return doc.replace("/", "__") + ".webp"


def code_hash(code: str) -> str:
    return hashlib.sha256((RENDERER_VERSION + code).encode()).hexdigest()[:16]


# --------------------------------------------------------------------------
# Running a page's plot code
# --------------------------------------------------------------------------

def capture_figure(code: str, cwd: Path):
    """Execute plot code and return the last plotly figure it shows."""
    import plotly.graph_objects as go
    import plotly.io as pio

    shown = []
    orig_fig_show, orig_pio_show = go.Figure.show, pio.show
    go.Figure.show = lambda self, *a, **k: shown.append(self)
    pio.show = lambda fig, *a, **k: shown.append(fig)
    old_cwd = os.getcwd()
    try:
        os.chdir(cwd)
        with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
            warnings.simplefilter("ignore")
            exec(compile(code, "<plot>", "exec"), {"__name__": "__gallery__"})
    finally:
        os.chdir(old_cwd)
        go.Figure.show, pio.show = orig_fig_show, orig_pio_show
    return shown[-1] if shown else None


# --------------------------------------------------------------------------
# plotly figure -> matplotlib thumbnail
# --------------------------------------------------------------------------

_RGB_RE = re.compile(r"rgba?\(([^)]*)\)")


def mpl_color(c, default=None):
    """Convert a plotly color string to something matplotlib accepts."""
    import matplotlib.colors as mcolors

    if c is None or not isinstance(c, str):
        return default
    m = _RGB_RE.fullmatch(c.replace(" ", ""))
    if m:
        parts = [float(p) for p in m.group(1).split(",")]
        rgb = [p / 255 for p in parts[:3]]
        return tuple(rgb + ([parts[3]] if len(parts) > 3 else []))
    return c if mcolors.is_color_like(c) else default


def _xy(trace):
    y = None if trace.y is None else np.asarray(trace.y)
    x = None if trace.x is None else np.asarray(trace.x)
    if y is not None and x is None:
        x = np.arange(len(y))
    if x is not None and x.dtype.kind in "OUS":
        try:
            x = np.asarray(x, dtype="datetime64[ns]")
        except (ValueError, TypeError):
            x = np.arange(len(x))
    if y is not None and y.dtype.kind == "O":
        y = np.asarray(y, dtype=float)
    return x, y


def _axis_key(ref: str, kind: str) -> str:
    """'x2' -> 'xaxis2', 'y' -> 'yaxis'."""
    suffix = (ref or kind)[1:]
    return f"{kind}axis{suffix}"


def render_thumbnail(fig, out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    layout = fig.layout.to_plotly_json()
    dpi = 100
    mfig = plt.figure(figsize=(THUMB_W * 2 / dpi, THUMB_H * 2 / dpi), dpi=dpi)
    mfig.patch.set_facecolor("white")

    axes: dict[tuple[str, str], object] = {}

    def domain(key, default=(0.0, 1.0)):
        return tuple((layout.get(key) or {}).get("domain") or default)

    def get_ax(xref, yref):
        xkey, ykey = _axis_key(xref, "x"), _axis_key(yref, "y")
        if (xkey, ykey) in axes:
            return axes[(xkey, ykey)]
        overlay = (layout.get(ykey) or {}).get("overlaying")
        if overlay and overlay != "free":
            base = get_ax(xref, overlay)
            ax = base.twinx()
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
        else:
            x0, x1 = domain(xkey)
            y0, y1 = domain(ykey)
            pad_l, pad_r, pad_b, pad_t = 0.03, 0.03, 0.04, 0.04
            w = 1 - pad_l - pad_r
            h = 1 - pad_b - pad_t
            gap = 0.012
            ax = mfig.add_axes([pad_l + x0 * w + gap, pad_b + y0 * h + gap,
                                (x1 - x0) * w - 2 * gap, (y1 - y0) * h - 2 * gap])
            ax.set_facecolor("#E5ECF6")
            ax.grid(True, color="white", lw=1.2)
            ax.set_axisbelow(True)
            ax.tick_params(left=False, bottom=False, labelleft=False,
                           labelbottom=False)
            for s in ax.spines.values():
                s.set_visible(False)
        yl = layout.get(ykey) or {}
        if yl.get("type") == "log":
            ax.set_yscale("log")
        axes[(xkey, ykey)] = ax
        return ax

    prev_y = {}
    categories: dict[int, list] = {}
    color_i = 0
    for trace in fig.data:
        if trace.visible in (False, "legendonly"):
            continue
        ttype = trace.type
        ax = get_ax(getattr(trace, "xaxis", None), getattr(trace, "yaxis", None))
        default = COLORWAY[color_i % len(COLORWAY)]
        color_i += 1
        opacity = trace.opacity if trace.opacity is not None else 1.0

        if ttype in ("scatter", "scattergl"):
            x, y = _xy(trace)
            if y is None:
                continue
            line = trace.line or {}
            marker = trace.marker or {}
            lc = mpl_color(line.color, None)
            mc = marker.color
            color = lc or mpl_color(mc if isinstance(mc, str) else None, default)
            mode = trace.mode or ("lines" if len(y) > 20 else "lines+markers")
            fill = trace.fill
            if fill in ("tozeroy", "tonexty"):
                base = prev_y.get(id(ax)) if fill == "tonexty" else None
                base = 0 if base is None or len(base[1]) != len(y) else base[1]
                fc = mpl_color(trace.fillcolor, None)
                ax.fill_between(x, base, y, color=fc or color,
                                alpha=1.0 if fc and len(fc) == 4 else 0.35,
                                lw=0, step="post" if line.shape == "hv" else None)
            if "lines" in mode and line.width != 0:
                lw = 1.6 * (line.width or 2) / 2
                kwargs = dict(color=color, lw=lw, alpha=opacity,
                              ls=DASHES.get(line.dash, "-"))
                if line.shape == "hv":
                    ax.step(x, y, where="post", **kwargs)
                else:
                    ax.plot(x, y, **kwargs)
            if "markers" in mode:
                size = (marker.size if isinstance(marker.size, (int, float)) else 6)
                if not isinstance(mc, str) and mc is not None:
                    try:
                        cvals = np.asarray(mc, dtype=float)
                        ax.scatter(x, y, c=cvals, s=size * 2.2, cmap="coolwarm",
                                   alpha=opacity, lw=0)
                        prev_y[id(ax)] = (x, y)
                        continue
                    except (TypeError, ValueError):
                        pass
                ax.scatter(x, y, color=mpl_color(mc if isinstance(mc, str) else None,
                                                 color),
                           s=size * 2.2, alpha=opacity, lw=0, zorder=3)
            prev_y[id(ax)] = (x, y)

        elif ttype == "bar":
            x, y = _xy(trace)
            if y is None:
                continue
            mc = (trace.marker or {}).color
            if isinstance(mc, str):
                colors = mpl_color(mc, default)
            elif mc is not None:
                colors = [mpl_color(c, default) for c in mc]
            else:
                colors = default
            raw_x = trace.x
            if raw_x is not None and len(raw_x) and isinstance(raw_x[0], str):
                # Categorical axis: place each label once, shared across traces.
                cats = categories.setdefault(id(ax), [])
                for lab in raw_x:
                    if lab not in cats:
                        cats.append(lab)
                x = np.array([cats.index(lab) for lab in raw_x])
            if trace.orientation == "h":
                ax.barh(np.arange(len(x)), np.asarray(x, dtype=float), color=colors)
            else:
                ax.bar(x, y, color=colors, alpha=opacity)

        elif ttype == "histogram":
            data = np.asarray(trace.x if trace.x is not None else trace.y, dtype=float)
            data = data[np.isfinite(data)]
            mc = mpl_color((trace.marker or {}).color, default)
            ax.hist(data, bins=trace.nbinsx or 30, color=mc, alpha=opacity)

        elif ttype in ("candlestick", "ohlc"):
            o, h, l, c = (np.asarray(v, dtype=float)
                          for v in (trace.open, trace.high, trace.low, trace.close))
            x = np.arange(len(c)) if trace.x is None else np.asarray(trace.x)
            up = c >= o
            ax.vlines(x, l, h, color="0.35", lw=0.5)
            ax.vlines(x[up], o[up], c[up], color="#2ca02c", lw=2)
            ax.vlines(x[~up], o[~up], c[~up], color="#d62728", lw=2)

        elif ttype == "heatmap":
            z = np.asarray(trace.z, dtype=float)
            ax.imshow(z, aspect="auto", origin="lower", cmap="viridis")

    # Reference lines from add_hline / add_vline.
    for shape in layout.get("shapes") or []:
        if shape.get("type", "line") != "line":
            continue
        xref = shape.get("xref", "x")
        yref = shape.get("yref", "y")
        xr, yr = xref.split(" ")[0], yref.split(" ")[0]
        if xr == "paper" or yr == "paper":
            continue
        ax = axes.get((_axis_key(xr, "x"), _axis_key(yr, "y")))
        if ax is None:
            continue
        lc = mpl_color((shape.get("line") or {}).get("color"), "0.4")
        ls = DASHES.get((shape.get("line") or {}).get("dash"), "-")
        if xref.endswith("domain") and shape.get("y0") == shape.get("y1"):
            ax.axhline(shape["y0"], color=lc, ls=ls, lw=1)
        elif yref.endswith("domain") and shape.get("x0") == shape.get("x1"):
            ax.axvline(shape["x0"], color=lc, ls=ls, lw=1)

    # Fixed ranges set in the layout.
    for (xkey, ykey), ax in axes.items():
        xr = (layout.get(xkey) or {}).get("range")
        yr = (layout.get(ykey) or {}).get("range")
        with contextlib.suppress(Exception):
            if xr and None not in xr:
                ax.set_xlim(*xr)
            if yr and None not in yr:
                if (layout.get(ykey) or {}).get("type") == "log":
                    yr = [10 ** v for v in yr]
                ax.set_ylim(*yr)

    buf = io.BytesIO()
    mfig.savefig(buf, format="png", dpi=dpi)
    plt.close(mfig)
    buf.seek(0)
    img = Image.open(buf).convert("RGB").resize((THUMB_W, THUMB_H), Image.LANCZOS)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path, format="WEBP", quality=80, method=6)


# --------------------------------------------------------------------------
# Gallery page
# --------------------------------------------------------------------------

def write_gallery(pages: list[dict], have_thumb: set[str]) -> None:
    groups = load_groups()
    topic_to_group = {t: g for g, e in groups.items() for t in e["topics"]}
    by_group: dict[str, list[dict]] = {g: [] for g in groups}
    for p in pages:
        if p["doc"] not in have_thumb:
            continue
        # A page appears under every group its topics belong to, as in the
        # function index.
        for g in dict.fromkeys(topic_to_group[t] for t in p["topics"]
                               if t in topic_to_group):
            by_group[g].append(p)

    lines = [
        ".. NOTE: This file is auto-generated by devtools/build_gallery.py",
        ".. Do not edit by hand. Re-run the script after changing a plot.",
        "",
        "Gallery",
        "=======",
        "",
        "One thumbnail per function page, grouped like the :doc:`function index",
        "<by_group_index>`. Each thumbnail is a static sketch of the page's",
        "interactive example; click it to open the page.",
        "",
    ]
    for gslug, entry in groups.items():
        members = sorted(by_group[gslug], key=lambda p: p["name"].lower())
        if not members:
            continue
        lines += [entry["name"], "-" * len(entry["name"]), "",
                  entry["desc"], "",
                  ".. grid:: 2 3 4 4", "   :gutter: 2", "   :class-container: gallery-grid", ""]
        for p in members:
            lines += [
                f"   .. grid-item-card:: {p['name']}",
                f"      :link: {p['doc']}",
                "      :link-type: doc",
                f"      :img-top: _static/gallery/{thumb_name(p['doc'])}",
                f"      :img-alt: {p['name']} example plot",
                "      :class-card: gallery-card",
                "",
            ]
    GALLERY_RST.write_text("\n".join(lines).rstrip() + "\n")


# --------------------------------------------------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--force", action="store_true", help="redraw every thumbnail")
    ap.add_argument("--only", help="regex on the page path; redraw exactly the matches")
    args = ap.parse_args(argv)

    manifest = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}
    fn_pages = function_pages()
    new_manifest, have_thumb, failed = {}, set(), []
    redrawn = 0
    for p in fn_pages:
        h = code_hash(p["code"])
        out = OUT_DIR / thumb_name(p["doc"])
        fresh = manifest.get(p["doc"]) == h and out.exists()
        wanted = args.force or not fresh
        if args.only:
            wanted = bool(re.search(args.only, str(p["path"])))
        if wanted:
            try:
                fig = capture_figure(p["code"], p["path"].parent)
                if fig is None:
                    raise RuntimeError("the code shows no figure")
                render_thumbnail(fig, out)
                redrawn += 1
            except Exception as exc:  # report and keep going
                failed.append((p["doc"], f"{type(exc).__name__}: {exc}"))
                continue
        if out.exists():
            new_manifest[p["doc"]] = h if wanted else manifest.get(p["doc"], h)
            have_thumb.add(p["doc"])

    # Drop thumbnails of pages that no longer exist or lost their plot.
    keep = {thumb_name(d) for d in new_manifest}
    for f in OUT_DIR.glob("*.webp"):
        if f.name not in keep:
            f.unlink()

    MANIFEST.write_text(json.dumps(dict(sorted(new_manifest.items())), indent=1) + "\n")
    write_gallery(fn_pages, have_thumb)
    print(f"{len(have_thumb)} thumbnails ({redrawn} redrawn), "
          f"{len(failed)} failed; wrote {GALLERY_RST.relative_to(ROOT)}")
    for doc, err in failed:
        print(f"  FAILED {doc}: {err}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
