"""Write wasm/smoke/session_oracle.json: live sessions that Python saved
midway, for the JavaScript suite to continue.

For each graph of tests/test_save_pipeline.py (the graphs of
devtools/pipeline_io_cases.py and a few joins), Python pushes the first part
of the events in the test's "blocked" order (one input runs ahead, so joins
hold events) and saves the session. The fixture holds the file (its library
version written as 0.0.0), the events before and after the save, and the
outputs after the second part and a flush.

js/test/save_pipeline.test.ts loads each file, pushes the second half and
checks the outputs; it also runs the first half itself and checks that its
file is Python's. NaN values are written as null (JSON has no NaN), which a
JavaScript push reads as NaN.

Modes:
    (no args)  write the fixture
    --stdout   print the fixture to stdout (freshness gate)
"""
import argparse
import json
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
OUT = os.path.join(ROOT, "wasm", "smoke", "session_oracle.json")
sys.path.insert(0, ROOT)


def plain(a):
    """An array as nested lists, NaN as None."""
    a = np.asarray(a)
    if a.dtype.kind == "f":
        return np.where(np.isnan(a), None, a).tolist()
    return a.tolist()


def outputs(result):
    """A session result as a list of {values, index}."""
    if isinstance(result, tuple) and len(result) == 2 and isinstance(result[0], np.ndarray) \
            and not isinstance(result[1], tuple):
        result = (result,)
    return [{"values": plain(v), "index": plain(i)} for v, i in result]


def events(evs):
    return [[k, t, plain(v)] for k, t, v in evs]


def compute():
    import screamer
    from tests.test_save_pipeline import CASES, _events, _push
    out = []
    for name, (pipe, feeds) in sorted(CASES.items()):
        first, rest = _events(feeds, "blocked")
        s = pipe.live()
        try:
            _push(s, first)
            text = s.dumps()
            b = screamer.loads(text)
            _push(b, rest)
            b.flush()
            result = b.result()
        except (ValueError, RuntimeError, TypeError):
            continue
        out.append({
            "name": name,
            "inputs": [n.op[1] for n in pipe.inputs],
            "first": events(first),
            "rest": events(rest),
            "file": re.sub(r'"library_version": "[^"]*"', '"library_version": "0.0.0"', text, count=1),
            "outputs": outputs(result),
        })
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stdout", action="store_true", help="print the fixture to stdout")
    args = ap.parse_args()
    text = json.dumps(compute(), separators=(",", ":"), allow_nan=False) + "\n"
    if args.stdout:
        sys.stdout.write(text)
        return
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"wrote {os.path.normpath(OUT)}")


if __name__ == "__main__":
    main()
