// A bundler follows the package's import of its WebAssembly module: esbuild
// bundles the entry with no copy step and no eval, and the bundle runs.
import { test } from "node:test";
import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { copyFileSync, existsSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";

const JS = fileURLToPath(new URL("..", import.meta.url));

test("bundle: esbuild includes the WASM module, and the bundle runs", async () => {
  const esbuild = await import("esbuild");
  const dir = mkdtempSync(join(tmpdir(), "screamer-bundle-"));
  try {
    const entry = join(dir, "app.mjs");
    writeFileSync(entry, `import { ready, RollingMean } from ${JSON.stringify(join(JS, "src", "index.ts"))};
await ready();
console.log(JSON.stringify(Array.from(RollingMean(2)(new Float64Array([1, 2, 3])))));
`);
    const out = join(dir, "bundle.mjs");
    await esbuild.build({ entryPoints: [entry], bundle: true, format: "esm", platform: "neutral",
                          mainFields: ["module", "main"], outfile: out, logLevel: "silent" });
    // A browser build resolves too (the glue's Node-only import is left to run time).
    await esbuild.build({ entryPoints: [entry], bundle: true, format: "esm", platform: "browser",
                          outfile: join(dir, "browser.mjs"), logLevel: "silent" });
    const code = readFileSync(out, "utf8");
    assert.ok(!code.includes("new Function"), "the bundle needs no eval");
    // The separate-file build loads screamer.wasm next to the module; the
    // single-file build embeds it.
    const wasm = join(JS, "src", "generated", "screamer.wasm");
    if (code.includes('new URL("screamer.wasm"') && existsSync(wasm)) copyFileSync(wasm, join(dir, "screamer.wasm"));
    const got = execFileSync(process.execPath, [out], { encoding: "utf8" }).trim();
    assert.equal(got, JSON.stringify([null, 1.5, 2.5]));
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});
