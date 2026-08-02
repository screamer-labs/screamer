import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import { ready } from "../src/index.ts";
import type { ScreamerOp } from "../src/index.ts";
import * as api from "../src/index.ts";

// Phase 2's Python oracle: a handful of ops driven event-by-event, with the
// output at one checkpoint recorded from the reference Python implementation.
// This proves the *ergonomic* op(...) call path (factories + wrapOp) reproduces
// that same output, i.e. the ergonomic layer does not diverge from the raw
// Embind substrate that wasm/smoke/smoke.mjs already validated.
const HERE = dirname(fileURLToPath(import.meta.url));
const REPO = join(HERE, "..", "..");
const oracle: Array<{
  name: string;
  args: Array<number | string | null>;
  inputs: number[][];
  check_index: number;
  expect: Array<number | null>;
}> = JSON.parse(readFileSync(join(REPO, "wasm", "smoke", "oracle.json"), "utf8"));

const TOL = 1e-9;

function closeEnough(got: number, expect: number): boolean {
  if (Number.isNaN(expect)) return Number.isNaN(got);
  return Math.abs(got - expect) <= TOL;
}

test("parity: oracle.json is non-trivial", async () => {
  await ready();
  assert.ok(oracle.length > 0);
});

for (const e of oracle) {
  test(`parity: ${e.name} matches Python oracle at check_index=${e.check_index}`, async () => {
    await ready();
    const factory = (api as any)[e.name] as (...args: any[]) => ScreamerOp;
    assert.ok(typeof factory === "function", `no exported factory named ${e.name}`);

    // JSON null == NaN sentinel for a missing optional ctor slot.
    const args = e.args.map((a) => (a === null ? NaN : a));
    const op = factory(...args);
    op.reset();

    let got: number | number[] | null = null;
    for (let i = 0; i < e.inputs.length; i++) {
      const tuple = e.inputs[i];
      const out = tuple.length === 1 ? op(tuple[0]) : op(...tuple);
      if (i === e.check_index) got = out;
    }
    op.dispose();

    assert.ok(got !== null, "check_index never reached");
    const gotArr = Array.isArray(got) ? got : [got as number];
    const expect = e.expect.map((v) => (v === null ? NaN : v));
    assert.equal(gotArr.length, expect.length, `${e.name}: output arity mismatch`);
    for (let k = 0; k < expect.length; k++) {
      assert.ok(
        closeEnough(gotArr[k], expect[k]),
        `${e.name}[${k}]: got=${gotArr[k]} expected=${expect[k]}`,
      );
    }
  });
}
