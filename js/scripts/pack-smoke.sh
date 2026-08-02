#!/usr/bin/env bash
# Pack-and-install smoke test: proves the PUBLISHED tarball is self-contained
# and computes correctly with zero consumer-side wasm wiring (no separate
# .wasm asset to resolve, no build step required on the install side).
#
# Steps: build the dist, `npm pack` it into a .tgz (this also re-runs the
# "prepack" lifecycle script, which rebuilds dist against the single-file
# wasm module), install that tgz into a throwaway project, then run a tiny
# script against the *installed* package (not the source tree).
set -euo pipefail

JS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$JS_DIR"

echo "== npm run build =="
npm run build

echo "== npm pack =="
# Don't capture `npm pack`'s stdout to get the tarball name: the "prepack"
# lifecycle script it triggers (rebuilding the single-file wasm) prints its
# own build log to stdout too, and that noise would corrupt a `$(npm pack)`
# capture. Instead let both print straight through, then find the .tgz on
# disk (clear out any stale ones first so the glob is unambiguous).
rm -f "$JS_DIR"/*.tgz
npm pack
TARBALL="$(ls -t "$JS_DIR"/*.tgz | head -n1)"
echo "packed: $TARBALL"

TMPDIR="$(mktemp -d "${TMPDIR:-/tmp}/screamer-pack-test.XXXXXX")"
cleanup() {
  rm -rf "$TMPDIR"
  rm -f "$TARBALL"
}
trap cleanup EXIT

echo "== install into $TMPDIR =="
(
  cd "$TMPDIR"
  npm init -y >/dev/null
  npm install "$TARBALL" >/dev/null
)

cat > "$TMPDIR/check.mjs" <<'EOF'
import assert from "node:assert/strict";
import { ready, RollingMean, Input, Pipeline, Diff } from "@screamer-labs/screamer";

await ready();

// A plain eager call: RollingMean(3)([1,2,3,4,5]) ~ [NaN, NaN, 2, 3, 4].
const out = RollingMean(3)([1, 2, 3, 4, 5]);
const expected = [NaN, NaN, 2, 3, 4];
assert.equal(out.length, expected.length, "RollingMean: output length");
for (let i = 0; i < expected.length; i++) {
  if (Number.isNaN(expected[i])) {
    assert.ok(Number.isNaN(out[i]), `RollingMean[${i}]: expected NaN, got ${out[i]}`);
  } else {
    assert.ok(
      Math.abs(out[i] - expected[i]) < 1e-9,
      `RollingMean[${i}]: expected ${expected[i]}, got ${out[i]}`,
    );
  }
}

// A define-then-bind Pipeline must also run end to end.
const x = Input("x");
const p = new Pipeline([x], [Diff(1)(RollingMean(3)(x))]);
const result = p([1, 2, 3, 4, 5, 6]);
assert.ok(result && result.values && result.values.length === 6, "Pipeline: unexpected output shape");

console.log("PACK OK");
EOF

node "$TMPDIR/check.mjs"
