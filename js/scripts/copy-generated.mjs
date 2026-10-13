// Copy the Emscripten module into a build output directory, with the
// screamer.wasm it loads when it has one.
//
//   node scripts/copy-generated.mjs dist/generated
//
// `npm run build:wasm` writes src/generated/screamer.mjs and the matching
// screamer.wasm side by side; the .mjs then loads "screamer.wasm" next to
// itself, so the .wasm must ship with it. `npm run build:wasm:single` writes a
// screamer.mjs that embeds its wasm and leaves an older screamer.wasm behind;
// that one must not ship, so it is removed from the output. A separate-file
// .mjs whose .wasm is missing or was written more than a minute apart from it
// (a stale pair) fails the build.
import { copyFileSync, existsSync, mkdirSync, readFileSync, rmSync, statSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const JS = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const SRC = join(JS, "src", "generated");
const dest = resolve(JS, process.argv[2] ?? "dist/generated");
const mjs = join(SRC, "screamer.mjs");
const wasm = join(SRC, "screamer.wasm");

mkdirSync(dest, { recursive: true });
copyFileSync(mjs, join(dest, "screamer.mjs"));
const separate = readFileSync(mjs, "utf8").includes('new URL("screamer.wasm"');
if (separate) {
  if (!existsSync(wasm)) {
    console.error(`copy-generated: ${mjs} loads screamer.wasm, but ${wasm} is missing; run npm run build:wasm`);
    process.exit(1);
  }
  const gap = Math.abs(statSync(mjs).mtimeMs - statSync(wasm).mtimeMs);
  if (gap > 60_000) {
    console.error(`copy-generated: screamer.mjs and screamer.wasm in ${SRC} were written ${Math.round(gap / 1000)} s apart, so they may not match; run npm run build:wasm`);
    process.exit(1);
  }
  copyFileSync(wasm, join(dest, "screamer.wasm"));
} else {
  rmSync(join(dest, "screamer.wasm"), { force: true });
}
console.log(`copy-generated: ${dest} <- screamer.mjs${separate ? " + screamer.wasm" : " (wasm embedded)"}`);
