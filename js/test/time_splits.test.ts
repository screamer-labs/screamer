import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import * as S from "../src/index.ts";

// timeSplits against Python (devtools/wasm/gen_time_splits_oracle.py writes
// the fixture; tests/test_time_splits_oracle_fresh.py keeps it current). Both
// run the one C++ function, so the folds must be equal.
const HERE = dirname(fileURLToPath(import.meta.url));
interface Fixture {
  cases: { n: number; nSplits: number; options: Record<string, number | number[]>; folds: [number[], number[]][] }[];
}
const fx: Fixture = JSON.parse(
  readFileSync(join(HERE, "..", "..", "wasm", "smoke", "oracle_time_splits.json"), "utf8"));

for (const c of fx.cases) {
  test(`timeSplits: n=${c.n} nSplits=${c.nSplits} ${Object.keys(c.options).join(",")} matches Python`, async () => {
    await S.ready();
    const got = S.timeSplits(c.n, c.nSplits, c.options as S.TimeSplitsOptions);
    assert.equal(got.length, c.folds.length);
    got.forEach(([train, test], f) => {
      assert.ok(train instanceof Int32Array && test instanceof Int32Array);
      assert.deepEqual(Array.from(train), c.folds[f][0], `fold ${f} train`);
      assert.deepEqual(Array.from(test), c.folds[f][1], `fold ${f} test`);
    });
  });
}

test("timeSplits: defaults to five folds and a label length of 0", async () => {
  await S.ready();
  const folds = S.timeSplits(12);
  assert.equal(folds.length, 5);
  assert.deepEqual(Array.from(folds[4][0]), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
  assert.deepEqual(Array.from(folds[4][1]), [10, 11]);
});

test("timeSplits: an NdArray label length equals the array", async () => {
  await S.ready();
  const label = Float64Array.from({ length: 20 }, (_, i) => (i === 8 ? 2 : 0));
  const a = S.timeSplits(20, 2, { testSize: 5, labelLength: label });
  const b = S.timeSplits(20, 2, { testSize: 5, labelLength: { data: label, shape: [20] } });
  assert.deepEqual(a.map(([tr]) => Array.from(tr)), b.map(([tr]) => Array.from(tr)));
  assert.deepEqual(Array.from(a[0][0]), [0, 1, 2, 3, 4, 5, 6, 7, 9]);
});

test("timeSplits: bad arguments throw naming the function", async () => {
  await S.ready();
  const nanLabel = new Float64Array(20);
  nanLabel[17] = NaN;
  const negLabel = new Float64Array(20);
  negLabel[4] = -1;
  const cases: [() => unknown, RegExp, new (...a: any[]) => Error][] = [
    [() => S.timeSplits(20, 2, { labelLength: nanLabel }), /^timeSplits: labelLength\[17\] is NaN; drop the rows without a label first, for example with forecastPairs\(\.\.\., \{ dropna: true \}\)$/, RangeError],
    [() => S.timeSplits(20, 2, { labelLength: NaN }), /^timeSplits: labelLength is NaN/, RangeError],
    [() => S.timeSplits(20, 2, { labelLength: negLabel }), /^timeSplits: labelLength\[4\] must not be negative/, RangeError],
    [() => S.timeSplits(20, 2, { labelLength: -1 }), /^timeSplits: labelLength must not be negative/, RangeError],
    [() => S.timeSplits(20, 2, { labelLength: new Float64Array(19) }),
      /^timeSplits: labelLength has 19 values, expected n = 20/, RangeError],
    [() => S.timeSplits(20, 2, { labelLength: { data: new Float64Array(20), shape: [20, 1] } }),
      /^timeSplits: labelLength must be a number or a 1-D array/, RangeError],
    [() => S.timeSplits(20, 2, { testSize: 5, labelLength: 10 }), /^timeSplits: .*no training rows/, RangeError],
    [() => S.timeSplits(5, 5), /^timeSplits: nSplits = 5 needs at least 6 rows, got n = 5/, RangeError],
    [() => S.timeSplits(20, 4, { testSize: 5 }),
      /^timeSplits: nSplits = 4 test blocks of 5 rows need more than n = 20 rows/, RangeError],
    [() => S.timeSplits(10, 1), /^timeSplits: nSplits must be an integer of at least 2/, RangeError],
    [() => S.timeSplits(10.5), /^timeSplits: n must be an integer of at least 1/, RangeError],
    [() => S.timeSplits(10, 2, { testSize: 0 }), /^timeSplits: testSize must be an integer of at least 1/, RangeError],
    [() => S.timeSplits(10, 2, { maxTrainSize: 0 }), /^timeSplits: maxTrainSize must be an integer of at least 1/, RangeError],
    [() => S.timeSplits(true as any), /^timeSplits: n must be an integer, got boolean$/, TypeError],
    [() => S.timeSplits(10, 2, { labelLength: true as any }), /^timeSplits: labelLength must be a number, got boolean$/, TypeError],
    [() => S.timeSplits(10, 2, { labelLength: "3" as any }), /^timeSplits: labelLength must be a number or an array/, TypeError],
  ];
  for (const [call, msg, kind] of cases) {
    assert.throws(call, (e: unknown) => e instanceof kind && msg.test((e as Error).message), String(msg));
  }
});
