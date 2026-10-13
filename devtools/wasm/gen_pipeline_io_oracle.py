"""Write wasm/smoke/pipeline_io_oracle.json: the Pipeline graphs of
devtools/pipeline_io_cases.py as Python saves them (to_json), with their
feeds and the results Python gives.

js/test/pipeline_io.test.ts loads each text with Pipeline.fromJSON, runs it
on the feeds and checks the results, and checks that its toJSON describes
the same graph. NaN values are written as null (JSON has no NaN), which a
JavaScript feed reads as NaN.

Modes:
    (no args)  write the fixture
    --stdout   print the fixture to stdout (freshness gate)
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
OUT = os.path.join(ROOT, "wasm", "smoke", "pipeline_io_oracle.json")
sys.path.insert(0, ROOT)


def plain(a):
    """An array as nested lists, NaN as None."""
    a = np.asarray(a)
    if a.dtype.kind == "f":
        return np.where(np.isnan(a), None, a).tolist()
    return a.tolist()


def outputs(result):
    """A Pipeline result as a list of {values, index}."""
    if isinstance(result, tuple) and len(result) == 2 and isinstance(result[0], np.ndarray) \
            and not isinstance(result[1], tuple):
        result = (result,)
    return [{"values": plain(v), "index": plain(i)} for v, i in result]


def compute():
    from devtools.pipeline_io_cases import cases
    out = []
    for name, (pipe, feeds) in sorted(cases().items()):
        out.append({
            "name": name,
            "json": pipe.to_json(),
            "inputs": [n.op[1] for n in pipe.inputs],
            "feeds": [{"values": plain(v), "index": plain(i)} for v, i in feeds],
            "outputs": outputs(pipe(*feeds)),
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
    with open(OUT, "w") as f:
        f.write(text)
    print(f"wrote {os.path.normpath(OUT)}")


if __name__ == "__main__":
    main()
