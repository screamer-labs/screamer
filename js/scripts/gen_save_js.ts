// Writes wasm/smoke/save_js.json: JavaScript's save files for every operator
// of wasm/smoke/save_oracle.json, after the same events Python fed.
// tests/test_save_restore.py::test_js_files_load_byte_for_byte loads them in
// Python and writes them back byte for byte; js/test/save_restore.test.ts
// checks that the committed file is what this script writes today.
//
//   node --import tsx scripts/gen_save_js.ts
import { readFileSync, writeFileSync } from "node:fs";
import { pathToFileURL } from "node:url";
import * as S from "../src/index.ts";

export const ORACLE_PATH = new URL("../../wasm/smoke/save_oracle.json", import.meta.url);
export const FIXTURE_PATH = new URL("../../wasm/smoke/save_js.json", import.meta.url);

const num = (v: number | null | string): number => (v === null ? NaN : typeof v === "string" ? Number(v) : v);
const decode = (x: any): any => (Array.isArray(x) ? x.map(decode)
  : x === null || (typeof x === "string" && /Infinity/.test(x)) ? num(x) : x);
const hex = (b: Uint8Array) => Array.from(b, (v) => v.toString(16).padStart(2, "0")).join("");

/** The fixture's text: one {name, json_text, cbor_hex} per case of the oracle. */
export async function fixtureText(): Promise<string> {
  await S.ready();
  const oracle = JSON.parse(readFileSync(ORACLE_PATH, "utf8"));
  const out = oracle.cases.map((c: any) => {
    const op = S.fromSettings({ type: c.name, settings: decode(c.settings) });
    try {
      for (let t = 0; t < c.prefix[0].length; t++) op(...c.prefix.map((x: any[]) => num(x[t])));
      return { name: c.name, json_text: op.dumps(), cbor_hex: hex(op.dumps("cbor")) };
    } finally {
      op.dispose();
    }
  });
  return JSON.stringify(out, null, 1) + "\n";
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  writeFileSync(FIXTURE_PATH, await fixtureText());
  console.log(`wrote ${FIXTURE_PATH.pathname}`);
}
