import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import { ready, PortfolioReport } from "../src/index.ts";
import type { NdArray } from "../src/index.ts";

// Dynamic-width reducer parity. PortfolioReport folds a variable number of
// assets into one event, so it is the one op that cannot go through the
// fixed-width event ABI the rest of the binding uses. The fixture in
// wasm/smoke/oracle_reducer.json is a (time, assets, 4) backtest engine output
// reduced by the *Python* build, recorded both as a single batch call and as a
// row-at-a-time stream; these tests hold the JavaScript build to it.
const HERE = dirname(fileURLToPath(import.meta.url));
const REPO = join(HERE, "..", "..");
const fx: {
  name: string;
  shape: [number, number, number];
  input: Array<Array<Array<number | null>>>;
  batch: Array<Array<number | null>>;
  stream: Array<Array<number | null>>;
} = JSON.parse(readFileSync(join(REPO, "wasm", "smoke", "oracle_reducer.json"), "utf8"));

const TOL = 1e-9;
const [TIME, ASSETS, WIDTH] = fx.shape;

// JSON null is the NaN sentinel, as in wasm/smoke/oracle.json.
const num = (v: number | null) => (v === null ? NaN : v);
const expectRows = fx.batch.map((r) => r.map(num));
const streamRows = fx.stream.map((r) => r.map(num));
const nested = fx.input.map((row) => row.map((group) => group.map(num)));

const flat = new Float64Array(TIME * ASSETS * WIDTH);
{
  let k = 0;
  for (const row of nested) for (const group of row) for (const v of group) flat[k++] = v;
}
const ndInput = (): NdArray => ({ data: flat, shape: [TIME, ASSETS, WIDTH] });

function assertClose(got: ArrayLike<number>, want: number[], what: string) {
  assert.equal(got.length, want.length, `${what}: output arity`);
  for (let i = 0; i < want.length; i++) {
    if (Number.isNaN(want[i])) {
      assert.ok(Number.isNaN(got[i]), `${what}[${i}]: expected NaN, got ${got[i]}`);
    } else {
      assert.ok(
        Math.abs(got[i] - want[i]) <= TOL,
        `${what}[${i}]: got ${got[i]}, expected ${want[i]}`,
      );
    }
  }
}

function assertMatrix(got: NdArray, want: number[][], what: string) {
  assert.deepEqual(got.shape, [want.length, want[0].length], `${what}: shape`);
  for (let r = 0; r < want.length; r++) {
    assertClose(got.data.subarray(r * want[0].length, (r + 1) * want[0].length), want[r], `${what} row ${r}`);
  }
}

test("reducer: the fixture is non-trivial", async () => {
  await ready();
  assert.ok(TIME > 1 && ASSETS > 1);
  // The fixture must contain the NaN row, otherwise the ignore policy is untested.
  assert.ok(expectRows.some((r) => r.every((v) => Number.isNaN(v))), "no NaN row in fixture");
});

test("reducer: batch over an NdArray matches the Python oracle", async () => {
  await ready();
  const op = PortfolioReport();
  try {
    assertMatrix(op(ndInput()) as NdArray, expectRows, "batch");
  } finally {
    op.dispose();
  }
});

test("reducer: batch over nested arrays matches the Python oracle", async () => {
  await ready();
  const op = PortfolioReport();
  try {
    assertMatrix(op(nested) as NdArray, expectRows, "nested batch");
  } finally {
    op.dispose();
  }
});

test("reducer: per-event rows match the Python stream, and agree with batch", async () => {
  await ready();
  const op = PortfolioReport();
  try {
    for (let t = 0; t < TIME; t++) {
      assertClose(op(nested[t]) as number[], streamRows[t], `event ${t}`);
    }
  } finally {
    op.dispose();
  }
  assert.deepEqual(streamRows, expectRows, "Python batch and stream must already agree");
});

test("reducer: a flat event is read as (groups, width)", async () => {
  await ready();
  const shaped = PortfolioReport();
  const flatOp = PortfolioReport();
  try {
    for (let t = 0; t < TIME; t++) {
      const row = nested[t];
      const flatRow = new Float64Array(ASSETS * WIDTH);
      for (let a = 0; a < ASSETS; a++) flatRow.set(row[a], a * WIDTH);
      assertClose(flatOp(flatRow) as number[], shaped(row) as number[], `flat event ${t}`);
    }
  } finally {
    shaped.dispose();
    flatOp.dispose();
  }
});

test("reducer: an iterable of rows streams the same values", async () => {
  await ready();
  const op = PortfolioReport();
  try {
    const out = [...(op(nested[Symbol.iterator]()) as Generator<number[]>)];
    assert.equal(out.length, TIME);
    for (let t = 0; t < TIME; t++) assertClose(out[t], streamRows[t], `streamed event ${t}`);
  } finally {
    op.dispose();
  }
});

test("reducer: an async iterable of rows streams the same values", async () => {
  await ready();
  const op = PortfolioReport();
  async function* source() {
    for (const row of nested) yield row;
  }
  try {
    const out: number[][] = [];
    for await (const row of op(source()) as AsyncGenerator<number[]>) out.push(row);
    assert.equal(out.length, TIME);
    for (let t = 0; t < TIME; t++) assertClose(out[t], streamRows[t], `async event ${t}`);
  } finally {
    op.dispose();
  }
});

test("reducer: reset restarts the reduction", async () => {
  await ready();
  const op = PortfolioReport();
  try {
    const first = op(nested[0]) as number[];
    for (let t = 1; t < TIME; t++) op(nested[t]);
    op.reset();
    assertClose(op(nested[0]) as number[], first, "after reset");
  } finally {
    op.dispose();
  }
});

test("reducer: batch is independent of prior events", async () => {
  await ready();
  const op = PortfolioReport();
  try {
    op(nested[0]);
    op(nested[1]);
    assertMatrix(op(ndInput()) as NdArray, expectRows, "batch after events");
  } finally {
    op.dispose();
  }
});

test("reducer: a group of the wrong width is rejected", async () => {
  await ready();
  const op = PortfolioReport();
  try {
    assert.throws(() => op([[1, 2, 3]]), TypeError);
    assert.throws(() => op({ data: new Float64Array(6), shape: [2, 3] } as NdArray), TypeError);
    assert.throws(() => op(new Float64Array(6)), TypeError);
    assert.throws(() => op({ data: new Float64Array(12), shape: [1, 3, 4, 1] } as NdArray), TypeError);
  } finally {
    op.dispose();
  }
});

test("reducer: a changed group count is rejected, leaving no trace", async () => {
  await ready();
  // The op fixes its group count at the first event after a reset, so a later
  // change is an error rather than a silent redefinition of the portfolio.
  const reference = PortfolioReport();
  const probe = PortfolioReport();
  try {
    reference(nested[0]);
    probe(nested[0]);
    assert.throws(() => probe(nested[0].slice(0, ASSETS - 1)), RangeError);
    assertClose(
      probe(nested[1]) as number[],
      reference(nested[1]) as number[],
      "event after a rejected one",
    );
  } finally {
    reference.dispose();
    probe.dispose();
  }
});

test("reducer: an empty event is rejected", async () => {
  await ready();
  const op = PortfolioReport();
  try {
    assert.throws(() => op(new Float64Array(0)), TypeError);
  } finally {
    op.dispose();
  }
});

test("reducer: use after dispose is rejected", async () => {
  await ready();
  const op = PortfolioReport();
  op.dispose();
  assert.throws(() => op(nested[0]), /after dispose/);
  assert.throws(() => op.reset(), /after dispose/);
});
