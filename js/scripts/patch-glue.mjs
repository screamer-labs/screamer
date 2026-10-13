// Hides the Node-only import of the Emscripten glue from bundlers.
//
//   node js/scripts/patch-glue.mjs wasm/build/screamer.mjs   (build-wasm.sh runs it)
//
// The glue (built for web, worker and node) runs `await import("module")`
// only under Node, to make a require(). A bundler resolves every literal
// import() and fails on "module" for a browser build (esbuild: "Could not
// resolve"; webpack: "Can't resolve"), so the specifier becomes an
// expression, which bundlers leave to run time, as Node runs it. Idempotent.
import { readFileSync, writeFileSync } from "node:fs";

const file = process.argv[2] ?? "src/generated/screamer.mjs";
const src = readFileSync(file, "utf8");
const from = 'import("module")';
const to = 'import(["module"][0])';
if (!src.includes(from) && !src.includes(to)) {
  console.error(`patch-glue: ${file} has no ${from}; check the Emscripten glue`);
  process.exit(1);
}
writeFileSync(file, src.split(from).join(to));
console.log(`patch-glue: ${file}`);
