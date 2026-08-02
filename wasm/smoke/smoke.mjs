// Node smoke test for the screamer WASM build.
//
//   node wasm/smoke/smoke.mjs
//
// Part A (coverage): construct EVERY manifest op with canonical args and run one
// event through evalInto; assert no throw and finite-or-NaN outputs.
// Part B (parity): drive the recorded Python-oracle series through a sample of
// ops and assert the checked output matches within 1e-9.
//
// A construct/eval failure or a parity mismatch is a real finding and fails the run.

import init from "../build/screamer.mjs";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

const HERE = dirname(fileURLToPath(import.meta.url));
const REPO = join(HERE, "..", "..");
const manifest = JSON.parse(
  readFileSync(join(REPO, "devtools", "wasm", "wasm_manifest.json"), "utf8"),
);
const oracle = JSON.parse(readFileSync(join(HERE, "oracle.json"), "utf8"));

const TOL = 1e-9;

// Per-op canonical-arg overrides (WASM/JS ctor convention: positional, NaN for a
// missing optional slot). Needed where the canonical defaults violate a ctor
// invariant (fast<slow, cutoff ranges, enum-valued strings, "exactly one decay").
// Each is documented with the reason the naive canonical would throw.
const OVERRIDES = {
  // fast must be strictly less than slow
  ADOSC: [3, 10],
  MACD: [12, 26, 9],
  // fill string must be "touch" or "breach"
  BacktestL1Orders: [1.0, 1.0, "touch", 1.0, 1.0, 1.0, 1.0],
  BacktestL1TradesOrders: [1.0, 1.0, "touch", 1.0, 1.0, 1.0, 1.0],
  BacktestOHLCOrders: [1.0, 1.0, "touch", 1.0, 1.0, 1.0, 1.0],
  BacktestTradesOrders: [1.0, 1.0, "touch", 1.0, 1.0, 1.0, 1.0],
  // Butterworth cutoffs must satisfy 0 < low < high < 1
  ButterBandpass: [2, 0.1, 0.4],
  ButterBandstop: [2, 0.1, 0.4],
  ButterHighpass: [2, 0.25],
  // output enum must be "cleaned"/"flag"/"nan"
  Hampel: [3, 1.0, "cleaned", "strict"],
  ImpulseClip: [3, 1.0, "cleaned", "strict"],
  // decay must be in (0,1): (decay, alpha, mu)
  HawkesIntensity: [0.5, 1.0, 0.0],
  // window size must be at least 4
  HullMA: [4],
  // alpha must be in (0,1)
  RollingCVaR: [3, 0.5],
  // window_size must be >= 4 * min_scale; method must be "rs"
  RollingHurst: [40, 4, "rs"],
  // output enum must be "mrr"/"mean"/"relmean"/"std"
  RollingOU: [3, "mean", "strict"],
  // derivative order must be a valid small int
  RollingPoly1: [3, 0, "strict"],
  RollingPoly2: [3, 0, "strict"],
  // method must be "wilder" or "cutler"
  RollingRSI: [3, "wilder", "strict"],
  // lower must be strictly less than upper: (lower, upper, initial)
  SchmittTrigger: [0.0, 1.0, 0.0],
  // ew_optional "exactly one of period/cutoff must be provided" (period slot)
  RoofingFilter: [48, 10, NaN, NaN],
  SuperSmoother: [10, NaN],
  Decycler: [10, NaN],
  // int window + 2 optional bounds + output enum ("clipped") + start policy
  RollingSigmaClip: [20, NaN, NaN, "clipped", "strict"],
};

// Build canonical args for one manifest entry.
function canonicalArgs(entry) {
  if (OVERRIDES[entry.name]) return OVERRIDES[entry.name];
  const ctor = entry.ctor;
  const args = ctor.map((t) => {
    switch (t) {
      case "int":
        return 3;
      case "double":
        return 1.0;
      case "std::string":
        return "strict";
      case "std::optional<double>":
        return NaN; // missing optional slot
      case "std::vector<double>":
        return [1.0, 2.0, 3.0];
      default:
        throw new Error(`${entry.name}: unknown ctor type ${t}`);
    }
  });
  // ew_optional ops usually require "exactly one" decay parameter; the naive
  // all-NaN optional set throws. Provide a valid value in the LAST optional slot
  // (alpha for the Ew* / DEMA / TEMA family; the trailing double is untouched).
  if (entry.ctor_kind === "ew_optional") {
    let lastOpt = -1;
    for (let i = 0; i < ctor.length; i++) {
      if (ctor[i] === "std::optional<double>") lastOpt = i;
    }
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

function isFiniteOrNaN(v) {
  return typeof v === "number" && (Number.isFinite(v) || Number.isNaN(v));
}

async function main() {
  const M = await init();

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
      const nIn = op.nIn();
      const out = evalOne(M, op, new Array(nIn).fill(1.0));
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
