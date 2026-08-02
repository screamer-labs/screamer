import { test } from "node:test";
import assert from "node:assert/strict";
import { init } from "../src/index.ts";

test("loads the WASM module and exposes raw ops + heap helpers", async () => {
  const M = await init();
  assert.equal(typeof M.RollingMean, "function");
  assert.equal(typeof M.allocF64, "function");
});
