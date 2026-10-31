import { test } from "node:test";
import assert from "node:assert/strict";
import { ready, BacktestTradesOrders, BacktestL1Orders } from "../src/index.ts";

await ready();

// Same cases as tests/test_backtest_cost_reference.py: a resting fill's cost is its fee alone.
test("backtest cost: a resting trades fill costs its fee", () => {
  const out = BacktestTradesOrders(0.001)([99, 99], [1, 1], [NaN, NaN], [0, 0],
                                          [101, 98], [5, 5]) as number[][];
  assert.equal(out[1][2], 1);                           // position after the fill
  assert.ok(Math.abs(out[1][3] - 0.099) < 1e-12);       // cost = 99 * 0.001
  assert.ok(Math.abs(out[1][0] - (-1 - 0.099)) < 1e-12); // equity unchanged: marked at 98
});

test("backtest cost: a resting L1 bid run over by the ask costs its fee", () => {
  const out = BacktestL1Orders(0.001)([99, 99], [1, 1], [NaN, NaN], [0, 0],
                                      [99.5, 97], [100.5, 98], [5, 5], [5, 5]) as number[][];
  assert.equal(out[1][2], 1);
  assert.ok(Math.abs(out[1][3] - 0.099) < 1e-12);
  assert.ok(Math.abs(out[1][0] - (-1.5 - 0.099)) < 1e-12);  // marked at the mid 97.5
});
