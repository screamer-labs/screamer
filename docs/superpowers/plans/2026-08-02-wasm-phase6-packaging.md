# screamer.js Phase 6 (prep): Packaging - Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `@screamer-labs/screamer` a genuinely installable, self-contained npm package - ESM + CJS, the WASM shipped and loading from the published tarball, npm metadata + README, a WASM size pass, and a tag-gated CI publish pipeline - WITHOUT publishing or bumping any version (those are a separate, user-gated step).

**Architecture:** The published package ships a self-contained single-file WASM module (Emscripten `-sSINGLE_FILE=1`, the `.wasm` embedded as base64 in the `.mjs`) so there is no asset-resolution problem for consumers. `tsc` emits ESM + declarations; a CJS build is produced alongside. A `pack`-and-install smoke proves the tarball works. A GitHub Actions workflow builds + tests + (on a version tag only) publishes to npm.

**Tech Stack:** Emscripten (single-file build), TypeScript (tsc), Node, npm, GitHub Actions.

## Global Constraints

- Do NOT publish to npm and do NOT edit any version file in this plan. Packaging only. The actual publish + shared version bump is a later, explicitly user-approved step (like the PyPI 2.0.0 release).
- Do NOT change the Python binding, C++ compute, the pure engine, or the Phase 2-5 generators/runtime/DAG. Packaging changes live under `js/` and `.github/workflows/`.
- The published package must be self-contained: `npm pack` then install the tarball into a fresh directory and `import`/`require` it must work with no extra asset wiring.
- The Python suite stays 6933/2. No em-dashes; no version file edits.

## File Structure

- Modify: `wasm/build-wasm.sh` - add a `--single-file` variant emitting `screamer.single.mjs` (`-sSINGLE_FILE=1`, `--closure=1`).
- Modify: `js/package.json` - `exports` map (import/require/types), `files`, metadata, a `prepack` that builds wasm + dist.
- Create: `js/tsconfig.cjs.json` - CJS emit config.
- Modify: `js/src/loader.ts` - import the single-file module so dist is self-contained.
- Create: `js/README.md` - usage.
- Create: `js/test/pack.test.mjs` (or a shell) - `npm pack` + install-elsewhere smoke.
- Create: `.github/workflows/js-publish.yml` - build + test + tag-gated npm publish.

---

### Task 1: Self-contained single-file WASM + dist that works from a tarball

**Files:**
- Modify: `wasm/build-wasm.sh` (a `--single-file` output), `js/package.json` (build wiring, `files`), `js/src/loader.ts` (import the single-file module)
- Create: `js/scripts/pack-smoke.sh`

**Interfaces:**
- Produces: `js/dist/` containing `index.js` (ESM) + `index.d.ts` + the embedded single-file WASM module, such that installing the packed tarball and doing `await ready(); RollingMean(3)(...)` + a `Pipeline` works.

- [ ] **Step 1: single-file WASM build**

In `wasm/build-wasm.sh`, add a build that emits a single-file ES module (wasm embedded as base64) with closure on:
`emcc ... -sSINGLE_FILE=1 --closure=1 -o "$BUILD/screamer.single.mjs"` (same sources/flags as the normal build otherwise). Print its size.

- [ ] **Step 2: package build wiring**

`js/package.json` scripts: `"build:wasm:single": "bash ../wasm/build-wasm.sh --single-file && cp ../wasm/build/screamer.single.mjs src/generated/screamer.mjs"` (so `loader.ts`'s import resolves to the self-contained module), then `"prepack": "npm run build:wasm:single && npm run build"`, and `"files": ["dist"]`. The dist must bundle the single-file module: either (a) have `loader.ts` import `./generated/screamer.mjs` and include the built `.mjs` in `dist/generated/` via a copy step in `build`, or (b) inline via the bundler. Choose the approach that makes `dist` self-contained and adjust `.gitignore` so the SOURCE `src/generated/screamer.mjs` stays ignored but the DIST copy ships.

- [ ] **Step 3: pack-and-install smoke**

`js/scripts/pack-smoke.sh`: `npm run build` (via prepack) then `npm pack` -> a `.tgz`; `mkdir /tmp/screamer-pack-test && cd` it, `npm init -y`, `npm install <abs path to tgz>`, write a tiny `check.mjs` that `import { ready, RollingMean, Input, Pipeline, Diff } from "@screamer-labs/screamer"; await ready(); assert RollingMean(3)([1,2,3,4,5]) ~ [NaN,NaN,2,3,4]; const x=Input("x"); const p=new Pipeline([x],[Diff(1)(RollingMean(3)(x))]); assert p([1,2,3,4,5,6]) works`, and run it. Print `PACK OK`.

- [ ] **Step 4: run it**

Run: `cd js && bash scripts/pack-smoke.sh`
Expected: `PACK OK` - the installed tarball computes correctly with no extra wiring. A missing-wasm or resolution error is a real finding (the dist is not self-contained; fix the build wiring).

- [ ] **Step 5: Commit**

```bash
git add wasm/build-wasm.sh js/package.json js/src/loader.ts js/scripts/pack-smoke.sh js/.gitignore
git commit -m "build(js): self-contained single-file WASM package; pack-and-install smoke"
```

---

### Task 2: CJS build + exports map

**Files:**
- Create: `js/tsconfig.cjs.json`
- Modify: `js/package.json` (exports map, build:cjs)

**Interfaces:**
- Produces: `dist/index.cjs` (+ `.d.cts` or shared `.d.ts`) so `require("@screamer-labs/screamer")` works alongside `import`.

- [ ] **Step 1: CJS tsconfig**

`js/tsconfig.cjs.json`: extends the base, `"module": "CommonJS"`, `"outDir": "dist/cjs"` (or emit `index.cjs` via a rename step). Note: the single-file WASM `.mjs` is ESM; for CJS a dynamic `import()` in the loader bridges it (`async function init(){ const m = await import("./generated/screamer.mjs"); ... }` already async, works from CJS via dynamic import). Verify the loader's dynamic import works under CJS.

- [ ] **Step 2: exports map**

`js/package.json`:
```json
"exports": { ".": { "types": "./dist/index.d.ts", "import": "./dist/index.js", "require": "./dist/cjs/index.cjs" } },
"main": "./dist/cjs/index.cjs", "module": "./dist/index.js", "types": "./dist/index.d.ts"
```
`"build": "npm run build:esm && npm run build:cjs"`.

- [ ] **Step 3: verify both entrypoints**

Run: `cd js && npm run build && node -e "require('./dist/cjs/index.cjs').ready" && node --input-type=module -e "import('./dist/index.js').then(m=>console.log(typeof m.ready))"`
Expected: both resolve `ready` (function). Extend `pack-smoke.sh` to also `require()` the package from a CJS file.

- [ ] **Step 4: Commit**

```bash
git add js/tsconfig.cjs.json js/package.json js/scripts/pack-smoke.sh
git commit -m "build(js): dual ESM + CJS output with an exports map"
```

---

### Task 3: npm metadata + README

**Files:**
- Modify: `js/package.json` (metadata)
- Create: `js/README.md`

- [ ] **Step 1: metadata**

`js/package.json`: `description`, `keywords` (streaming, time-series, technical-analysis, wasm, indicators), `license` (match the repo's license), `repository` (`git+https://github.com/screamer-labs/screamer.git`, `directory: "js"`), `homepage`, `bugs`, `author`, `sideEffects: false`, `engines.node`.

- [ ] **Step 2: README**

`js/README.md`: install (`npm i @screamer-labs/screamer`), the `await ready()` note, the four-regime `RollingMean(3)(...)` example, the Pipeline example, a note that it is the JS build of the Python `screamer` package with matching semantics, and a link back. Follow the repo writing style (no em-dashes, lead with the point).

- [ ] **Step 3: verify metadata**

Run: `cd js && npm pkg get name version description license repository.url` and `npm publish --dry-run 2>&1 | tail -20`
Expected: the dry-run lists the tarball contents (dist + README) with no errors and no unexpected files (no src, no node_modules, no build artifacts beyond dist). Do NOT actually publish.

- [ ] **Step 4: Commit**

```bash
git add js/package.json js/README.md
git commit -m "docs(js): npm metadata + README"
```

---

### Task 4: WASM size pass

**Files:**
- Modify: `wasm/build-wasm.sh` (closure + any safe size flags)

- [ ] **Step 1: measure + apply size levers**

The DAG build is ~104 KB gzip. Apply and measure: `--closure=1` (JS glue), `-Oz` (already), `-sFILESYSTEM=0` (already), and try `-sMINIMAL_RUNTIME` / `-sTEXTDECODER=2` / `-sABORTING_MALLOC=0` where safe (each must keep the full JS + parity suites green). Record before/after gzip sizes for the single-file build in the report. Do NOT sacrifice correctness for size (re-run `node --import tsx --test test/*.test.ts` after each flag change).

- [ ] **Step 2: verify no regression**

Run: `cd js && npm run build:wasm:single && node --import tsx --test test/*.test.ts`
Expected: all pass; report the new size. If a flag breaks a test, drop it.

- [ ] **Step 3: Commit**

```bash
git add wasm/build-wasm.sh
git commit -m "build(wasm): size pass (closure + safe emcc flags); record sizes"
```

---

### Task 5: CI publish pipeline (defined, tag-gated, not triggered)

**Files:**
- Create: `.github/workflows/js-publish.yml`

- [ ] **Step 1: workflow**

`.github/workflows/js-publish.yml`: on `push` tags `js-v*` (a JS-specific tag so it does not fire on the Python `v*` tags) and `workflow_dispatch`. Jobs: (1) set up emsdk (actions or a cached install) + Node; build the WASM (`bash wasm/build-wasm.sh --single-file`); `cd js && npm ci && npm run build`; run `node --import tsx --test test/*.test.ts` and the Python-oracle-dependent tests it can (or skip the oracle regen in CI and rely on the committed oracles); run the `pack-smoke.sh`. (2) A `publish` job gated on the tag: `npm publish --provenance --access public` using an `NPM_TOKEN` secret (or OIDC trusted publishing if configured). Mark the publish job clearly as the release gate.

- [ ] **Step 2: lint the workflow**

Run: `cd /Users/thijs/screamer && python3 -c "import yaml; yaml.safe_load(open('.github/workflows/js-publish.yml')); print('workflow yaml OK')"`
Expected: `workflow yaml OK`. (The workflow is defined but will not run until a `js-v*` tag is pushed, which is the user-gated release step - out of scope here.)

- [ ] **Step 3: Commit**

```bash
git add .github/workflows/js-publish.yml
git commit -m "ci(js): tag-gated npm publish pipeline (build + test + pack, publish on js-v tag)"
```

---

## Notes for the executor

- Task 1 is the crux: the package MUST work when installed from its tarball with no consumer-side wasm wiring. The single-file (`-sSINGLE_FILE=1`) build is the robust default; the pack-and-install smoke is the gate. If `dist` ends up missing the wasm module, the dist build wiring is wrong - fix it, do not ship a broken package.
- Do NOT publish and do NOT bump the version. The workflow's publish job and the version bump are the user's explicit release step.
- Keep the source `src/generated/screamer.mjs`/`.wasm` gitignored; only the `dist` output ships (via `files: ["dist"]`).
- Everything is under `js/`, `wasm/build-wasm.sh`, and `.github/workflows/`. No Python/C++/engine/Phase-2-5 code changes.
