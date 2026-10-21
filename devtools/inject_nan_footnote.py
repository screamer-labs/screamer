#!/usr/bin/env python3
"""Inject the NaN-policy footnote into every function help page.

Idempotent. For each ``docs/functions_*/<Name>.md``:

1. Read the YAML frontmatter. If it already declares ``nan_policy``, use that
   value. Otherwise, look the function up in :data:`POLICY_TABLE` and write
   ``nan_policy: <value>`` into the frontmatter. If the function is missing
   from both the frontmatter and the table, abort: that means a new function
   was added without a policy declaration and the contract in
   ``docs/nan_policy.md`` is silently violated.
2. Replace any existing ``<!-- NAN_FOOTNOTE_START --> ... <!-- NAN_FOOTNOTE_END -->``
   block in the body with a freshly-rendered footnote chosen by the policy.
   If no sentinel block exists yet, insert one immediately before
   ``## Examples`` (or, failing that, ``<!-- HELP_END -->``, or end-of-body).
   Placement guarantees the footnote ends up in the JSON ``details`` field
   produced by ``build_help_registry.py``.

Run:
    poetry run python devtools/inject_nan_footnote.py

The frontmatter and footnote together are what the contract relies on:
external frontends read the structured ``nan_policy`` field; humans read the
rendered footnote.
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"

ALLOWED_POLICIES = ("ignore", "propagate", "nan-aware")

# Bootstrap classification. Once every function file carries a nan_policy
# field in its frontmatter, this table is only consulted when a new function
# is added without one, and we deliberately fail in that case rather than
# guess. The frontmatter, not this table, is the source of truth.
POLICY_TABLE: dict[str, str] = {
    # functions_ew/ (all ignore)
    "EwBeta": "ignore",
    "EwCorr": "ignore",
    "EwCov": "ignore",
    "EwGarmanKlassVar": "ignore",
    "EwGarmanKlassVol": "ignore",
    "EwKurt": "ignore",
    "EwMean": "ignore",
    "EwParkinsonVar": "ignore",
    "EwParkinsonVol": "ignore",
    "EwRms": "ignore",
    "EwRogersSatchellVar": "ignore",
    "EwRogersSatchellVol": "ignore",
    "EwSkew": "ignore",
    "EwStd": "ignore",
    "EwVar": "ignore",
    "EwZscore": "ignore",
    # functions_fin/: positional return-types are propagate, rest ignore
    "Drawdown": "ignore",
    "LogReturn": "propagate",
    "MaxDrawdown": "ignore",
    "ROC": "propagate",
    "ROCP": "propagate",
    "ROCR": "propagate",
    "Return": "propagate",
    "RollingAlpha": "ignore",
    "RollingBeta": "ignore",
    "RollingCalmar": "ignore",
    "RollingCorr": "ignore",
    "RollingCov": "ignore",
    "RollingHitRate": "ignore",
    "RollingInfoRatio": "ignore",
    "RollingLinearRegression": "ignore",
    "RollingMaxDrawdown": "ignore",
    "RollingPercentile": "ignore",
    "RollingRank": "ignore",
    "RollingResidualStd": "ignore",
    "RollingSharpe": "ignore",
    "RollingSortino": "ignore",
    "RollingSpread": "ignore",
    "RollingTSF": "ignore",
    # functions_math/ (all ignore; stateless, IEEE arithmetic does the work)
    "Abs": "ignore",
    "Acos": "ignore",
    "Asin": "ignore",
    "Atan": "ignore",
    "Atan2": "ignore",
    "Cart2Polar": "ignore",
    "Ceil": "ignore",
    "Cos": "ignore",
    "Cube": "ignore",
    "Elu": "ignore",
    "Erf": "ignore",
    "Erfc": "ignore",
    "Exp": "ignore",
    "Floor": "ignore",
    "Hypot": "ignore",
    "Linear": "ignore",
    "Linear2": "ignore",
    "Log": "ignore",
    "Polar2Cart": "ignore",
    "Power": "ignore",
    "Relu": "ignore",
    "Round": "ignore",
    "Selu": "ignore",
    "Sigmoid": "ignore",
    "Sign": "ignore",
    "Sin": "ignore",
    "Softsign": "ignore",
    "Sqrt": "ignore",
    "Square": "ignore",
    "Tanh": "ignore",
    # functions_misc/: positional offsets are propagate
    "CumMax": "ignore",
    "CumMin": "ignore",
    "CumProd": "ignore",
    "CumSum": "ignore",
    "Detrend": "ignore",
    "Diff": "propagate",
    "Diff2": "propagate",
    "Identity": "ignore",
    "Lag": "propagate",
    "Momentum": "propagate",
    # functions_preprocessing/: FillNa and Ffill are NaN-aware by design
    "Clip": "ignore",
    "Ffill": "nan-aware",
    "FillNa": "nan-aware",
    # functions_rolling/ (all ignore)
    "AD": "ignore",
    "ADOSC": "ignore",
    "ADX": "ignore",
    "ATR": "ignore",
    "BOP": "ignore",
    "BollingerBands": "ignore",
    "CCI": "ignore",
    "DEMA": "ignore",
    "DonchianChannels": "ignore",
    "HullMA": "ignore",
    "KAMA": "ignore",
    "KeltnerChannels": "ignore",
    "MACD": "ignore",
    "MFI": "ignore",
    "NATR": "ignore",
    "OBV": "ignore",
    "RollingArgmax": "ignore",
    "RollingArgmin": "ignore",
    "RollingGarmanKlassVar": "ignore",
    "RollingGarmanKlassVol": "ignore",
    "RollingHurst": "ignore",
    "RollingIqr": "ignore",
    "RollingKurt": "ignore",
    "RollingMad": "ignore",
    "RollingMax": "ignore",
    "RollingMean": "ignore",
    "RollingMedian": "ignore",
    "RollingMin": "ignore",
    "RollingMinMax": "ignore",
    "RollingOU": "ignore",
    "RollingParkinsonVar": "ignore",
    "RollingParkinsonVol": "ignore",
    "RollingPoly1": "ignore",
    "RollingPoly2": "ignore",
    "RollingQuantile": "ignore",
    "RollingRSI": "ignore",
    "RollingRange": "ignore",
    "RollingRms": "ignore",
    "RollingRogersSatchellVar": "ignore",
    "RollingRogersSatchellVol": "ignore",
    "RollingSigmaClip": "ignore",
    "RollingSkew": "ignore",
    "RollingStd": "ignore",
    "RollingSum": "ignore",
    "RollingVWAP": "ignore",
    "RollingVar": "ignore",
    "RollingYangZhangVar": "ignore",
    "RollingYangZhangVol": "ignore",
    "RollingZscore": "ignore",
    "Stoch": "ignore",
    "StochRSI": "ignore",
    "TEMA": "ignore",
    "TRIMA": "ignore",
    "TRIX": "ignore",
    "TrueRange": "ignore",
    "UltimateOscillator": "ignore",
    "WMA": "ignore",
    "WilliamsR": "ignore",
    # functions_signal/ (all ignore; IIR/FIR filters, observation skipping)
    "Butter": "ignore",
    "ButterBandpass": "ignore",
    "ButterBandstop": "ignore",
    "ButterHighpass": "ignore",
    "KalmanFilter": "ignore",
    "MovingAverage": "ignore",
}

NAN_PAGE_LINK = "See [NaN and warmup](../nan_and_warmup.md)."

FOOTNOTE_BODY = {
    "ignore": (
        "**Policy: `ignore`.** A `NaN` in any input gives a `NaN` output at that "
        "step. The step is skipped: it does not enter the window or update the "
        "internal state. " + NAN_PAGE_LINK
    ),
    # Stateless element-wise functions (implementation_family: math) have no
    # window or state to skip, so their footnote says only what reaches the
    # output.
    "ignore-stateless": (
        "**Policy: `ignore`.** A `NaN` in any input gives a `NaN` output at that "
        "step. " + NAN_PAGE_LINK
    ),
    # Cross-sectional functions (implementation_family: cross) work row by row
    # across assets, so a NaN removes one asset from that row, not the step.
    "ignore-cross": (
        "**Policy: `ignore`.** A `NaN` for an asset gives a `NaN` output for that "
        "asset at that step, and the statistic is computed across the remaining "
        "assets. " + NAN_PAGE_LINK
    ),
    # Cross-sectional reductions (one value per row) leave a NaN asset out of
    # the row; the output is NaN only when too few assets are left.
    "ignore-cross-reduce": (
        "**Policy: `ignore`.** A `NaN` for an asset leaves that asset out of the "
        "row, and the statistic is computed across the remaining assets. A row "
        "with too few valid assets gives `NaN`. " + NAN_PAGE_LINK
    ),
    # Backtest engines give a NaN field its own meaning (hold the position,
    # skip the bar, leave a quote side empty), so the footnote defers to the
    # Description rather than promising a NaN output row.
    "ignore-engine": (
        "**Policy: `ignore`.** A `NaN` input never corrupts the engine's state. "
        "What a `NaN` target, price or quote does on this engine is stated in the "
        "Description. " + NAN_PAGE_LINK
    ),
    "propagate": (
        "**Policy: `propagate`.** A `NaN` input is stored in the lookback like any "
        "other value. The output is `NaN` wherever the formula references it and "
        "recovers once it leaves the lookback. " + NAN_PAGE_LINK
    ),
    "nan-aware": (
        "**Policy: `nan-aware`.** This function is built to process `NaN` inputs, "
        "as the Description explains. " + NAN_PAGE_LINK
    ),
}

STATELESS_FAMILIES = ("math",)
CROSS_FAMILIES = ("cross",)
# Cross-sectional operators that reduce a row to one value instead of one per asset.
CROSS_REDUCTIONS = ("CrossStd",)
KIND_RE = re.compile(r"^kind:\s*(.+?)\s*$", re.M)
FAMILY_RE = re.compile(r"^implementation_family:\s*(.+?)\s*$", re.M)

START_MARKER = "<!-- NAN_FOOTNOTE_START -->"
END_MARKER = "<!-- NAN_FOOTNOTE_END -->"
HELP_END_MARKER = "<!-- HELP_END -->"

FRONTMATTER_RE = re.compile(r"\A(---\n)(.*?)(\n---\n)", re.DOTALL)
NAN_POLICY_RE = re.compile(r"^nan_policy:\s*(.+?)\s*$", re.M)
SENTINEL_BLOCK_RE = re.compile(
    re.escape(START_MARKER) + r".*?" + re.escape(END_MARKER),
    re.DOTALL,
)
EXAMPLES_H2_RE = re.compile(r"^## Examples\s*$", re.M)


def is_backtest_engine(name: str) -> bool:
    """Backtest engines (not the report/statistics functors that read them)."""
    return name.startswith("Backtest") and name not in ("BacktestReport", "BacktestStats")


def render_footnote(policy: str, family: str | None = None, name: str = "") -> str:
    """Return the full sentinel-wrapped footnote block for a policy."""
    key = policy
    if policy == "ignore" and is_backtest_engine(name):
        key = "ignore-engine"
    elif policy == "ignore" and family in STATELESS_FAMILIES:
        key = "ignore-stateless"
    elif policy == "ignore" and family in CROSS_FAMILIES:
        key = "ignore-cross-reduce" if name in CROSS_REDUCTIONS else "ignore-cross"
    body = FOOTNOTE_BODY[key]
    return (
        f"{START_MARKER}\n"
        f"## NaN handling\n\n"
        f"{body}\n"
        f"{END_MARKER}"
    )


def resolve_policy(name: str, fm_text: str) -> tuple[str, str | None]:
    """Find or assign the function's NaN policy.

    Returns (policy, new_frontmatter_text or None). If the frontmatter
    already declares nan_policy, new_frontmatter_text is None. Otherwise it
    is the frontmatter with the field appended.
    """
    m = NAN_POLICY_RE.search(fm_text)
    if m:
        policy = m.group(1).strip().strip("'\"")
        if policy not in ALLOWED_POLICIES:
            raise SystemExit(
                f"{name}: frontmatter declares nan_policy={policy!r}, "
                f"not in {ALLOWED_POLICIES}"
            )
        return policy, None

    if name not in POLICY_TABLE:
        raise SystemExit(
            f"{name}: no nan_policy in frontmatter and no entry in "
            f"POLICY_TABLE. New function? Add it to POLICY_TABLE in "
            f"devtools/inject_nan_footnote.py, or add the field to the "
            f"frontmatter directly."
        )
    policy = POLICY_TABLE[name]
    new_fm = fm_text.rstrip() + f"\nnan_policy: {policy}"
    return policy, new_fm


def inject_footnote(body: str, footnote: str) -> str:
    """Insert or replace the sentinel-bounded footnote block in the body."""
    if SENTINEL_BLOCK_RE.search(body):
        return SENTINEL_BLOCK_RE.sub(lambda _m: footnote, body)

    block = f"\n{footnote}\n\n"
    m = EXAMPLES_H2_RE.search(body)
    if m:
        return body[: m.start()] + block + body[m.start():]
    if HELP_END_MARKER in body:
        idx = body.index(HELP_END_MARKER)
        return body[:idx] + block + body[idx:]
    return body.rstrip() + "\n" + block


def process_file(path: Path) -> bool:
    """Update one file in-place. Returns True if the file was modified."""
    original = path.read_text()
    m = FRONTMATTER_RE.match(original)
    if not m:
        return False

    fm_open, fm_text, fm_close = m.group(1), m.group(2), m.group(3)
    kind = KIND_RE.search(fm_text)
    if kind and kind.group(1).strip() != "functor":
        return False  # stream classes, plain functions and guides have no policy
    body = original[m.end():]
    name = path.stem

    policy, new_fm = resolve_policy(name, fm_text)
    final_fm = new_fm if new_fm is not None else fm_text

    fam = FAMILY_RE.search(final_fm)
    family = fam.group(1).strip() if fam else None
    new_body = inject_footnote(body, render_footnote(policy, family, name))
    text = fm_open + final_fm + fm_close + new_body

    if text == original:
        return False
    path.write_text(text)
    return True


def main(argv: list[str]) -> int:
    files = sorted(DOCS.glob("functions_*/*.md"))
    wanted = set(argv) if argv else None
    if wanted:
        files = [p for p in files if p.stem in wanted]

    n_modified = 0
    for path in files:
        try:
            if process_file(path):
                n_modified += 1
                print(f"  + {path.relative_to(ROOT)}")
        except SystemExit as e:
            print(str(e), file=sys.stderr)
            return 1

    print(f"\nProcessed {len(files)} files, modified {n_modified}.")
    return 0


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("names", nargs="*", help="process only these pages (default: all)")
    sys.exit(main(ap.parse_args().names))
