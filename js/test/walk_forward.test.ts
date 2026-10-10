import { test } from "node:test";
import assert from "node:assert/strict";
import { ready, walkForward, BacktestOHLCTarget, BacktestPortfolioTarget } from "../src/index.ts";

await ready();

// The deterministic market and candidates of tests/test_walk_forward.py.
function market(T: number) {
  const o: number[] = [], h: number[] = [], l: number[] = [], c: number[] = [];
  let p = 100;
  for (let t = 0; t < T; t++) {
    const prev = p;
    p *= Math.exp(0.01 * Math.sin(1.7 * t) * Math.cos(0.31 * t * t));
    o.push(prev); c.push(p); h.push(Math.max(prev, p) * 1.001); l.push(Math.min(prev, p) * 0.999);
  }
  return [o, h, l, c];
}

function candidates(T: number): number[][] {
  const t = Array.from({ length: T }, (_, i) => i);
  return [t.map((i) => Math.sign(Math.sin(i / 6))), t.map((i) => Math.sign(Math.cos(i / 17))),
          t.map((i) => (Math.floor(i / 5) % 3) - 1)];
}

const near = (a: number, b: number) => Math.abs(a - b) <= 1e-12 * Math.max(1, Math.abs(b));

// Same golden values as tests/test_walk_forward.py::test_golden_values.
test("walkForward: golden values match Python", () => {
  const T = 260;
  const res = walkForward(BacktestOHLCTarget(2e-4), candidates(T), market(T),
    { train: 90, test: 40, periodsPerYear: 252 });
  assert.deepEqual(Array.from(res.foldStart), [90, 130, 170, 210, 250]);
  assert.deepEqual(Array.from(res.choice), [1, 1, 0, 2, 2]);
  assert.deepEqual(res.trainScores.shape, [5, 3]);
  const col0 = [0, 1, 2, 3, 4].map((i) => res.trainScores.data[i * 3]);
  [1.2084732360973462, 0.9425475505694672, 1.620956872474087,
   -1.0708210557093047, -2.150979534091671].forEach((v, i) => assert.ok(near(col0[i], v), `${col0[i]}`));
  assert.ok(near(res.stats.sharpe, -0.8521351650503325), `${res.stats.sharpe}`);
  assert.ok(near(res.stats.total_pnl, -4.302590266628476), `${res.stats.total_pnl}`);
  assert.equal(res.metric, "sharpe");
  assert.deepEqual(res.output.shape, [T, 4]);

  const anch = walkForward(BacktestOHLCTarget(2e-4), candidates(T), market(T),
    { train: 90, test: 40, anchored: true, metric: "total_pnl" });
  assert.deepEqual(Array.from(anch.choice), [1, 1, 1, 0, 0]);
});

test("walkForward: stitched target, flat folds, and one candidate", () => {
  const T = 300;
  const X = candidates(T);
  const late = X.map((x) => x.map((v, t) => (t < 150 ? NaN : v)));
  const res = walkForward(BacktestOHLCTarget(), late, market(T), { train: 100, test: 50 });
  assert.deepEqual(Array.from(res.choice).slice(0, 2), [-1, -1]);
  for (let t = 0; t < 200; t++) assert.equal(res.target[t], 0);
  res.foldStart.forEach((t0, i) => {
    const k = res.choice[i];
    for (let t = t0; t < Math.min(t0 + 50, T); t++)
      assert.equal(res.target[t], k < 0 ? 0 : late[k][t]);
  });
  const one = walkForward(BacktestOHLCTarget(), X[0], market(T), { train: 100, test: 50 });
  assert.ok(Array.from(one.choice).every((k) => k === 0));
});

test("walkForward: argument checks raise RangeError", () => {
  const T = 100, X = candidates(T), m = market(T);
  const eng = BacktestOHLCTarget();
  assert.throws(() => walkForward(eng, X, m, { train: 1, test: 10 }), RangeError);
  assert.throws(() => walkForward(eng, X, m, { train: 10, test: 0 }), RangeError);
  assert.throws(() => walkForward(eng, X, m, { train: 100, test: 10 }), RangeError);
  assert.throws(() => walkForward(eng, X, m, { train: 10, test: 10, metric: "bogus" as any }), RangeError);
  assert.throws(() => walkForward(eng, X, m, { train: 10, test: 10, metric: "turnover" }), RangeError);
  assert.throws(() => walkForward(eng, X, [m[0].slice(1), m[1], m[2], m[3]], { train: 10, test: 10 }),
    RangeError);
  assert.throws(() => walkForward(BacktestPortfolioTarget(), X, m, { train: 10, test: 10 }), RangeError);
});
