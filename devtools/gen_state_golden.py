"""Write the golden save files of operators (tests/data/state_golden).

For each operator named on the command line this runs its case of
tests/save_cases.py: it feeds the first events (two windows, or 50 events
without a window) one at a time and writes

* ``<Name>.v<N>.json``, the save file after those events, and
* ``<Name>.v<N>.case.json``, the events fed (``prefix``), the next events
  (``rest``) and the outputs they give (``rest_output``),

where N is the operator's current state version. tests/test_save_restore.py
checks that today's operator still writes the golden of its version, so a
changed state layout needs a new kStateVersion and a new golden. An existing
golden that differs is not overwritten unless --force is given.

Run from the repository root:

    python devtools/gen_state_golden.py RollingMean EwMean
"""
import argparse
import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tests.save_cases import case, feed, kwargs_for, splits  # noqa: E402

GOLDEN = ROOT / "tests" / "data" / "state_golden"
REST = 10


def golden(name):
    """(save file text, case JSON text, state version) of ``name``."""
    make, inputs = case(name)
    k = splits(name, kwargs_for(name))[2]
    op = make()
    feed(op, inputs, 0, k)
    text = op.dumps()
    version = op.state()["version"]
    rest = [x[k:k + REST] for x in inputs]
    scenario = {
        "prefix": [x[:k].tolist() for x in inputs],
        "rest": [x.tolist() for x in rest],
        "rest_output": feed(op, rest, 0, REST).tolist(),
    }
    return text, json.dumps(scenario) + "\n", version


def _layout(text):
    """The save file without its library version, which every release changes."""
    return re.sub(r'"library_version": "[^"]*"', '"library_version": ""', text, count=1)


def _write(path, text):
    with open(path, "w", encoding="utf-8", newline="") as fh:
        fh.write(text)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("names", nargs="+", help="operator names")
    parser.add_argument("--force", action="store_true",
                        help="overwrite a golden of the same state version that differs")
    args = parser.parse_args(argv)
    GOLDEN.mkdir(parents=True, exist_ok=True)
    failed = False
    for name in args.names:
        text, scenario, version = golden(name)
        path = GOLDEN / f"{name}.v{version}.json"
        if path.exists() and _layout(path.read_text(encoding="utf-8")) != _layout(text) \
                and not args.force:
            print(f"{name}: the state layout differs from {path.name}; bump kStateVersion "
                  "(or pass --force for a golden that has not been released)", file=sys.stderr)
            failed = True
            continue
        _write(path, text)
        _write(GOLDEN / f"{name}.v{version}.case.json", scenario)
        print(f"wrote {path.relative_to(ROOT)}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
