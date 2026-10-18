// Node smoke test for the screamer WASM build.
//
//   node wasm/smoke/smoke.mjs [path/to/screamer.mjs]
//
// Part A (coverage): construct EVERY manifest op with its default args (from the
// manifest) and run one event through evalInto (mapRowInto for a row op, with
// 3 assets); assert no throw and finite-or-NaN outputs.
// Part B (parity): drive the recorded Python-oracle series through a sample of
// ops and assert the checked output matches within 1e-9.
//
// A construct/eval failure or a parity mismatch is a real finding and fails the run.

import { readFileSync } from "node:fs";
import { fileURLToPath, pathToFileURL } from "node:url";
import { dirname, join, resolve } from "node:path";

// The module to test: wasm/build/screamer.mjs, or the path given as the first
// argument (`npm test` passes js/src/generated/screamer.mjs, the copy the
// package ships).
const { default: init } = await import(
  process.argv[2] ? pathToFileURL(resolve(process.argv[2])).href : "../build/screamer.mjs");

const HERE = dirname(fileURLToPath(import.meta.url));
const REPO = join(HERE, "..", "..");
const manifest = JSON.parse(
  readFileSync(join(REPO, "devtools", "wasm", "wasm_manifest.json"), "utf8"),
);
const oracle = JSON.parse(readFileSync(join(HERE, "oracle.json"), "utf8"));

const TOL = 1e-9;

// Constructor arguments come from the manifest's C++ defaults
// (devtools/wasm/gen_wasm_manifest.py records each `"name"_a = default`), so
// they follow the bindings and cannot go stale. A default the parser does not
// read raises, naming the op and the text, so a new default form gets a rule
// here instead of a silent wrong value. A missing default (none today) or an
// optional slot without one takes the canonical value of its type.
function parseDefault(name, type, text) {
  if (text === null || text === undefined) {
    if (type === "std::optional<double>") return NaN;     // a missing optional slot
    return { int: 3, double: 1.0, bool: true, "std::string": "strict", "std::vector<double>": [1.0] }[type];
  }
  const t = text.trim();
  let m;
  if ((m = /^"(.*)"$/.exec(t))) return m[1];
  if (type === "bool" && (t === "true" || t === "false")) return t === "true";
  if ((m = /^(-?)std::numeric_limits<double>::infinity\(\)$/.exec(t))) return m[1] ? -Infinity : Infinity;
  if ((m = /^screamer::PerAsset\{(.*)\}$/.exec(t)) || (m = /^std::vector<double>\{(.*)\}$/.exec(t)))
    return m[1].split(",").map((x) => Number(x.trim()));
  if (/^-?[0-9.]+(e-?[0-9]+)?$/i.test(t)) return Number(t);
  throw new Error(`${name}: cannot read the default ${JSON.stringify(text)} of a ${type} parameter`);
}

// Values for parameters that have no default and that the canonical value of
// their type does not satisfy, by parameter name. A name the constructor no
// longer has raises, so an entry cannot go stale silently.
const REQUIRED = {
  SchmittTrigger: { lower: 0.0, upper: 1.0 },         // lower < upper
  RoofingFilter: { hp_period: 48, lp_period: 10 },     // one of period or cutoff per filter
  RollingOLS: { window_size: 20 },                     // at least n_regressors + 2
};

// Build constructor args for one manifest entry from its defaults.
function canonicalArgs(entry) {
  const names = entry.ctor_args.map((a) => a.name);
  const args = entry.ctor.map((type, i) => parseDefault(entry.name, type, entry.ctor_args[i]?.default ?? null));
  for (const [param, value] of Object.entries(REQUIRED[entry.name] ?? {})) {
    const i = names.indexOf(param);
    if (i < 0) throw new Error(`REQUIRED names ${param}, which ${entry.name} no longer takes; update wasm/smoke/smoke.mjs`);
    args[i] = value;
  }
  // ew_optional ops require exactly one decay parameter and default to none:
  // set the last optional slot (alpha for the Ew* / DEMA / TEMA family).
  if (entry.ctor_kind === "ew_optional" && args.every((a, i) => entry.ctor[i] !== "std::optional<double>" || Number.isNaN(a))) {
    let lastOpt = -1;
    for (let i = 0; i < entry.ctor.length; i++) if (entry.ctor[i] === "std::optional<double>") lastOpt = i;
    if (lastOpt >= 0) args[lastOpt] = 0.5;
  }
  return args;
}

function construct(M, name, args) {
  const Cls = M[name];
  if (!Cls) throw new Error(`op ${name} not registered on module`);
  // Embind's register_vector needs an actual VectorDouble instance, not a JS
  // array. Materialize any array arg into one, construct, then free it (the C++
  // ctor copies the vector).
  const vecs = [];
  const materialized = args.map((a) => {
    if (Array.isArray(a)) {
      const v = new M.VectorDouble();
      for (const x of a) v.push_back(x);
      vecs.push(v);
      return v;
    }
    return a;
  });
  try {
    return new Cls(...materialized);
  } finally {
    for (const v of vecs) v.delete();
  }
}

// Run one event (all inputs = fill) through evalInto; return the output array.
function evalOne(M, op, inputs) {
  const nIn = op.nIn();
  const nOut = op.nOut();
  const inBuf = M.allocF64(nIn);
  const outBuf = M.allocF64(nOut);
  try {
    M.viewF64(inBuf, nIn).set(inputs);
    op.evalInto(inBuf, outBuf);
    return Array.from(M.viewF64(outBuf, nOut));
  } finally {
    M.freeBuf(inBuf);
    M.freeBuf(outBuf);
  }
}

// One event of a row op: fieldsIn() rows of k assets (1, 2, ...), field-major.
function evalRow(M, op, k) {
  const fi = op.fieldsIn(), fo = op.fieldsOut();
  const inBuf = M.allocF64(fi * k);
  const outBuf = M.allocF64(fo * k);
  try {
    M.viewF64(inBuf, fi * k).set(Array.from({ length: fi * k }, (_, i) => 1 + (i % k)));
    op.mapRowInto(inBuf, outBuf, k);
    return Array.from(M.viewF64(outBuf, fo * k));
  } finally {
    M.freeBuf(inBuf);
    M.freeBuf(outBuf);
  }
}

function isFiniteOrNaN(v) {
  return typeof v === "number" && (Number.isFinite(v) || Number.isNaN(v));
}

async function main() {
  const M = await init();
  // A heap view, as js/src/loader.ts makes it (the module exports no view).
  M.viewF64 = (p, n) => new Float64Array(M.HEAPF64.buffer, p, n);

  // ---- Part A: coverage ----
  let constructed = 0;
  const failures = [];
  const skips = [];
  for (const entry of manifest) {
    const name = entry.name;
    let op = null;
    try {
      const args = canonicalArgs(entry);
      op = construct(M, name, args);
      // A row op (variable width) maps one row of 3 assets per field.
      const out = entry.row_op ? evalRow(M, op, 3) : evalOne(M, op, new Array(op.nIn()).fill(1.0));
      for (const v of out) {
        if (!isFiniteOrNaN(v)) {
          throw new Error(`non-finite output: ${JSON.stringify(out)}`);
        }
      }
      constructed++;
    } catch (err) {
      failures.push(`${name}: ${err && err.message ? err.message : err}`);
    } finally {
      if (op && typeof op.delete === "function") op.delete();
    }
  }

  // ---- Part B: parity against the Python oracle ----
  let matched = 0;
  const mismatches = [];
  for (const e of oracle) {
    let op = null;
    try {
      // JSON null == NaN sentinel for a missing optional ctor slot.
      const args = e.args.map((a) => (a === null ? NaN : a));
      op = construct(M, e.name, args);
      op.reset();
      let got = null;
      for (let i = 0; i < e.inputs.length; i++) {
        const out = evalOne(M, op, e.inputs[i]);
        if (i === e.check_index) got = out;
      }
      const expect = e.expect.map((v) => (v === null ? NaN : v));
      let ok = got !== null && got.length === expect.length;
      if (ok) {
        for (let k = 0; k < expect.length; k++) {
          const g = got[k];
          const x = expect[k];
          const good = Number.isNaN(x) ? Number.isNaN(g) : Math.abs(g - x) <= TOL;
          if (!good) ok = false;
        }
      }
      if (ok) {
        matched++;
      } else {
        mismatches.push(
          `${e.name}: got=${JSON.stringify(got)} expected=${JSON.stringify(expect)}`,
        );
      }
    } catch (err) {
      mismatches.push(`${e.name}: THREW ${err && err.message ? err.message : err}`);
    } finally {
      if (op && typeof op.delete === "function") op.delete();
    }
  }

  // ---- Report ----
  if (skips.length) {
    console.log(`Skipped ${skips.length} op(s):`);
    for (const s of skips) console.log(`  SKIP ${s}`);
  }
  if (failures.length) {
    console.log(`\n${failures.length} coverage FAILURE(s):`);
    for (const f of failures) console.log(`  FAIL ${f}`);
  }
  if (mismatches.length) {
    console.log(`\n${mismatches.length} parity MISMATCH(es):`);
    for (const m of mismatches) console.log(`  MISMATCH ${m}`);
  }

  const total = manifest.length;
  if (failures.length || mismatches.length) {
    console.log(
      `\nSMOKE FAILED: ${constructed}/${total} ops constructed+evaluated, ` +
        `${matched}/${oracle.length} parity checks matched`,
    );
    process.exit(1);
  }

  console.log(
    `SMOKE OK: ${constructed} ops constructed+evaluated, ${matched} parity checks matched`,
  );
}

main().catch((err) => {
  console.error("smoke harness error:", err);
  process.exit(1);
});
