#!/usr/bin/env python3
"""Inject the NaN-policy footnote into every function help page.

Idempotent. For each ``docs/functions_*/<Name>.md``:

1. Read the YAML frontmatter. If it already declares ``nan_policy``, use that
   value. Otherwise, look the function up in :data:`POLICY_TABLE` and write
   ``nan_policy: <value>`` into the frontmatter. If the function is missing
   from both the frontmatter and the table, abort -- that means a new function
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
# is added without one -- and we deliberately fail in that case rather than
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
    # functions_fin/ -- positional return-types are propagate, rest ignore
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
    # functions_math/ (all ignore -- stateless, IEEE arithmetic does the work)
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
    # functions_misc/ -- positional offsets are propagate
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
    # functions_preprocessing/ -- FillNa and Ffill are NaN-aware by design
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
    # functions_signal/ (all ignore -- IIR/FIR filters, observation skipping)
    "Butter": "ignore",
    "ButterBandpass": "ignore",
    "ButterBandstop": "ignore",
    "ButterHighpass": "ignore",
    "KalmanFilter": "ignore",
    "MovingAverage": "ignore",
}

FOOTNOTE_BODY = {
    "ignore": (
        "**Policy: `ignore`.** A `NaN` in any input at index `t` causes the "
        "function to skip that step: output at `t` is `NaN` and internal state "
        "is unchanged. Subsequent finite samples are processed as if step `t` "
        "had not occurred."
    ),
    "propagate": (
        "**Policy: `propagate`.** Input `NaN` values are stored in the lookback. "
        "Output is `NaN` at any index where the function's positional formula "
        "references a `NaN` input; recovery happens once the `NaN` slides out "
        "of the lookback."
    ),
    "nan-aware": (
        "**Policy: `nan-aware`.** This function is designed to consume `NaN` "
        "inputs; see the description above for its specific behavior."
    ),
}

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


def render_footnote(policy: str) -> str:
    """Return the full sentinel-wrapped footnote block for a policy."""
    body = FOOTNOTE_BODY[policy]
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
    body = original[m.end():]
    name = path.stem

    policy, new_fm = resolve_policy(name, fm_text)
    final_fm = new_fm if new_fm is not None else fm_text

    new_body = inject_footnote(body, render_footnote(policy))
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
    sys.exit(main(sys.argv[1:]))
