import { test } from "node:test";
import assert from "node:assert/strict";
import {
  ready, CrossRank, CrossZscore, CrossWinsorize, CrossNormalize, CrossTopBottom, CrossDemean,
  CrossStd,
} from "../src/index.ts";

await ready();

const N = Number.NaN;
const X = [[3, 1, 2, N, -1], [1, 1, 1, 1, 1], [N, N, N, N, N], [5, -1, 2, 0, 2]];
const nd = { data: Float64Array.from(X.flat()), shape: [4, 5] };

// Same values as Python (screamer.CrossRank()(X) etc. on the same matrix).
const GOLDEN: [string, () => any, number[]][] = [
  ["rank", () => CrossRank(), [1.0, -0.33333333333333337, 0.33333333333333326, N, -1.0, 0, 0, 0, 0, 0,
    N, N, N, N, N, 1.0, -1.0, 0.25, -0.5, 0.25]],
  ["zscore", () => CrossZscore(), [1.02469507659596, -0.14638501094227999, 0.43915503282683993, N,
    -1.3174650984805198, 0, 0, 0, 0, 0, N, N, N, N, N, 1.4768656253944359, -1.1293678311839805,
    0.17374889710522773, -0.6949955884209111, 0.17374889710522773]],
  ["winsorize", () => CrossWinsorize(0.25, 0.75), [2.25, 1, 2, N, 0.5, 1, 1, 1, 1, 1, N, N, N, N, N,
    2, 0, 2, 0, 2]],
  ["normalize", () => CrossNormalize(2.0, 0.5), [0.875, -0.075, 0.375, N, -0.6749999999999999,
    0, 0, 0, 0, 0, N, N, N, N, N, 1.011904761904762, -0.4642857142857143, 0.11904761904761904,
    -0.28571428571428575, 0.11904761904761904]],
  ["topbottom", () => CrossTopBottom(2), [1, -1, 1, N, -1, 0, 0, 0, 0, 0, N, N, N, N, N,
    1, -1, 0, -1, 1]],
  ["demean", () => CrossDemean(), [1.75, -0.25, 0.75, N, -2.25, 0, 0, 0, 0, 0, N, N, N, N, N,
    3.4, -2.6, 0.3999999999999999, -1.6, 0.3999999999999999]],
];

const same = (a: ArrayLike<number>, b: ArrayLike<number>) =>
  a.length === b.length &&
  Array.from(a).every((v, i) => (Number.isNaN(v) && Number.isNaN(b[i])) || Math.abs(v - b[i]) <= 1e-12);

for (const [name, make, want] of GOLDEN) {
  test(`cross: ${name} matches Python in every regime`, () => {
    const batch = make()(nd);
    assert.deepEqual(batch.shape, [4, 5]);
    assert.ok(same(batch.data, want), `${name} batch: ${Array.from(batch.data)}`);
    assert.ok(same(make()(X).flatMap((r: Float64Array) => Array.from(r)), want), `${name} array of rows`);
    const op = make();
    assert.ok(same(X.flatMap((r) => Array.from(op(r) as number[])), want), `${name} row by row`);
    const streamed = [...make()(X.values())].flatMap((r: number[]) => r);
    assert.ok(same(streamed, want), `${name} stream`);
  });
}

test("cross: one row keeps its container kind; width is fixed until reset", () => {
  const op = CrossRank();
  assert.ok(op(Float64Array.of(1, 2, 3)) instanceof Float64Array);
  assert.ok(Array.isArray(op([1, 2, 3])));
  assert.throws(() => op([1, 2]));
  op.reset();
  assert.deepEqual(op([2, 1]), [1, -1]);
});

test("cross: argument checks", () => {
  assert.throws(() => CrossRank("percent"));
  assert.throws(() => CrossNormalize(1.0, 2.0));
  assert.throws(() => CrossTopBottom(0));
});

test("cross: a size-1 field axis is consumed (rule 3)", () => {
  for (const [name, make, want] of GOLDEN) {
    const unit = make()({ data: nd.data, shape: [4, 5, 1] });
    assert.deepEqual(unit.shape, [4, 5], name);
    assert.ok(same(unit.data, want), `${name} [T, K, 1]`);
    const events = X.map((r) => ({ data: Float64Array.from(r), shape: [5, 1] }));
    const streamed = [...make()(events.values())].flatMap((r: Float64Array) => Array.from(r));
    assert.ok(same(streamed, want), `${name} [K, 1] events`);
  }
});

test("A1: an array of rows returns an array of rows", () => {
  const out = CrossRank()([[1, 3, 2], [3, 1, 2]]) as Float64Array[];
  assert.ok(Array.isArray(out) && out.length === 2 && out[0] instanceof Float64Array);
  const ref = CrossRank()({ data: Float64Array.from([1, 3, 2, 3, 1, 2]), shape: [2, 3] }) as any;
  assert.deepEqual([...out[0], ...out[1]], [...ref.data]);
  assert.deepEqual(ref.shape, [2, 3]);
  const typed = CrossRank()([Float64Array.from([1, 3, 2]), Float64Array.from([3, 1, 2])]) as any;
  assert.ok(Array.isArray(typed) && typed[1] instanceof Float64Array);
});

// CrossStd is a row reduction: one value per row, the K axis drops out.
// Same values as Python (screamer.CrossStd(ddof)(X) on the same matrix).
const STD: [number, number[]][] = [
  [1, [1.707825127659933, 0.0, N, 2.3021728866442674]],
  [0, [1.479019945774904, 0.0, N, 2.0591260281974]],
];

for (const [ddof, want] of STD) {
  test(`cross: CrossStd(${ddof}) matches Python in every regime`, () => {
    const batch = CrossStd(ddof)(nd);
    assert.deepEqual(batch.shape, [4]);
    assert.ok(same(batch.data, want), `batch: ${Array.from(batch.data)}`);
    const rows = CrossStd(ddof)(X);
    assert.ok(Array.isArray(rows) && rows.every((v: unknown) => typeof v === "number"));
    assert.ok(same(rows, want), "array of rows");
    const op = CrossStd(ddof);
    const one = X.map((r) => op(r));
    assert.ok(one.every((v) => typeof v === "number"));
    assert.ok(same(one, want), "row by row");
    assert.ok(same([...CrossStd(ddof)(X.values())], want), "stream");
    const unit = CrossStd(ddof)({ data: nd.data, shape: [4, 5, 1] });
    assert.deepEqual(unit.shape, [4]);
    assert.ok(same(unit.data, want), "[T, K, 1]");
  });
}

test("cross: CrossStd defaults, NaN rule and arguments", () => {
  assert.ok(same(CrossStd()(nd).data, STD[0][1]));
  assert.ok(Number.isNaN(CrossStd()([N, 2, N])));
  assert.equal(CrossStd(0)([N, 2, N]), 0);
  assert.deepEqual(CrossStd()({ data: new Float64Array(0), shape: [0, 3] }).shape, [0]);
  assert.throws(() => CrossStd(-1));
});
