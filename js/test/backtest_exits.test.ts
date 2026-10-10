import { test } from "node:test";
import assert from "node:assert/strict";
import { ready, BacktestOHLCTarget, BacktestStats, BacktestReport } from "../src/index.ts";

await ready();

// Same case as tests/test_backtest.py::test_stop_loss_exits_at_level_inside_the_bar.
test("backtest exits: OHLC stop-loss exits at the level", () => {
  const o = [100, 100, 99], h = [100, 101, 100], l = [100, 99, 94], c = [100, 100, 96];
  const eng = BacktestOHLCTarget(0, 0, -Infinity, Infinity, 1, 0, 0, "fraction", "open", 1, 0.05, 0, 0);
  const out = eng([1, 1, 1], o, h, l, c) as { data: Float64Array; shape: number[] };
  assert.deepEqual([out.data[2], out.data[6], out.data[10]], [0, 1, 0]);
  assert.equal(out.data[8], -5);
});

// Same case as tests/test_backtest.py::test_backtest_stats_hand_computed.
test("backtest stats: hand-computed round trips and annualized ratios", () => {
  const pos = [0, 1, 1, 0, -1, 1, 1, 0], pnl = [0, 1, 2, -1, 3, -2, 1, 1];
  const eq: number[] = []; let acc = 0; for (const p of pnl) eq.push(acc += p);
  const cost = pos.map(() => 0);
  const st = BacktestStats(1)(eq, pnl, pos, cost) as { data: Float64Array; shape: number[] };
  const last = Array.from(st.data.slice(st.data.length - 9));
  assert.equal(last[0], 5 / 8);
  assert.equal(last[1], 3);
  assert.ok(Math.abs(last[6] - 5 / 3) < 1e-12);
  const r1 = BacktestReport(1)(eq, pnl, pos, cost) as { data: Float64Array };
  const r252 = BacktestReport(252)(eq, pnl, pos, cost) as { data: Float64Array };
  const n = r1.data.length;
  assert.ok(Math.abs(r252.data[n - 1] - r1.data[n - 1] * Math.sqrt(252)) < 1e-9);
});
