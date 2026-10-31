import { test } from "node:test";
import assert from "node:assert/strict";
import {
  ready, BacktestOHLCOrders, BacktestTradesOrders, BacktestL1Orders, BacktestL1TradesOrders,
} from "../src/index.ts";

await ready();

// Same cases as tests/test_backtest_crossing.py: an order that crosses the market on
// arrival fills as a taker at the market, capped at its limit. Fees: maker 0.001, taker 0.002.
const close = (a: number, b: number) => Math.abs(a - b) < 1e-12;

test("crossing: OHLC bid above the bar fills at the open as a taker", () => {
  const out = BacktestOHLCOrders(0.001, 0.002)([110], [1], [NaN], [0],
                                              [100], [105], [95], [100]) as number[][];
  assert.equal(out[0][2], 1);
  assert.ok(close(out[0][3], 0.2));            // taker fee on a fill at 100
});

test("crossing: OHLC gap through a resting bid fills at the open", () => {
  const out = BacktestOHLCOrders(0.001, 0.002)([100], [1], [NaN], [0],
                                              [95], [97], [93], [96]) as number[][];
  assert.ok(close(out[0][1], 96 - 95 - 0.19)); // bought at 95, closed at 96
});

test("crossing: Trades re-priced bid above the print takes at the print", () => {
  const out = BacktestTradesOrders(0.001, 0.002)([99, 102], [1, 1], [NaN, NaN], [0, 0],
                                                [100, 100], [1, 1]) as number[][];
  assert.equal(out[1][2], 1);
  assert.ok(close(out[1][3], 0.2));
});

test("crossing: L1 bid re-priced through the ask takes at the ask", () => {
  const out = BacktestL1Orders(0.001, 0.002)([99, 102], [1, 1], [NaN, NaN], [0, 0],
                                            [100, 100], [101, 101], [5, 5], [5, 5]) as number[][];
  assert.equal(out[1][2], 1);
  assert.ok(close(out[1][3], 0.5 + 0.202));    // 101 vs mid 100.5, plus the taker fee
});

test("crossing: L1Trades new bid at the ask is a taker", () => {
  const out = BacktestL1TradesOrders(0.001, 0.002)([101], [1], [NaN], [0], [100], [101],
                                                  [5], [5], [NaN], [NaN]) as number[][];
  assert.equal(out[0][2], 1);
  assert.ok(close(out[0][3], 0.5 + 0.202));
});
