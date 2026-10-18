import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { spawnSync } from "node:child_process";
import { fileURLToPath } from "node:url";
import * as screamer from "../src/index.ts";
import { formatShow, guide, search, show, topics } from "../src/docs.ts";

const oracle = JSON.parse(readFileSync(new URL("./docs_oracle.json", import.meta.url), "utf8")) as
  { query: string; topic: string | null; limit: number; names: string[] }[];

const allNames = (): string[] => {
  const names = new Set<string>();
  for (const g of topics()) for (const t of g.topics) for (const f of t.functions) names.add(f);
  return [...names];
};

test("docs: search ranks exactly as the Python search", () => {
  for (const { query, topic, limit, names } of oracle) {
    const got = search(query, { topic: topic ?? undefined, limit }).map((r) => r.name);
    assert.deepEqual(got, names, query);
  }
});

test("docs: every documented name is exported by the package", () => {
  for (const name of allNames()) {
    const e = show(name);
    if (e.kind === "guide") continue;
    assert.ok(name in screamer, `${name} is documented but not exported`);
  }
});

test("docs: every functor accepts its documented options with their defaults", async () => {
  await screamer.ready();
  const lib = screamer as unknown as Record<string, (opts: object) => { dispose?: () => void }>;
  let checked = 0;
  for (const name of allNames()) {
    const e = show(name);
    if (!e.parameters?.length) continue;
    const opts: Record<string, unknown> = {};
    for (const p of e.parameters)
      if (p.defaultText !== undefined) opts[p.name] = Function(`return (${p.defaultText})`)();
      else if (p.required) opts[p.name] = p.default;   // the docs example value
    const op = lib[name](opts);
    op.dispose?.();
    checked++;
  }
  assert.ok(checked >= 150, `only ${checked} functors checked`);
});

test("docs: show finds JavaScript and Python names, and suggests on a typo", () => {
  assert.equal(show("positionTest").pythonName, "position_test");
  assert.equal(show("position_test").name, "positionTest");
  assert.equal(show("CombineLatest").name, "combineLatest");
  assert.equal(show("rollingrsi").name, "RollingRSI");
  assert.deepEqual(show("RollingRSI").signatures,
    ['RollingRSI({ windowSize: 14, method: "wilder", startPolicy: "strict" })']);
  assert.throws(() => show("RollingMaen"), /did you mean RollingMean/);
  assert.match(formatShow(show("VPIN")), /^# VPIN\n/);
  assert.match(guide(), /^# Using screamer: a guide for coding agents/);
});

test("docs: the command line", () => {
  const cli = fileURLToPath(new URL("../src/docs_cli.ts", import.meta.url));
  const run = (...args: string[]) =>
    spawnSync(process.execPath, ["--import", "tsx", cli, ...args], { encoding: "utf8" });
  let r = run("search", "kalman", "--limit", "2");
  assert.equal(r.status, 0, r.stderr);
  assert.match(r.stdout, /^KalmanFilter +Scalar/);
  r = run("--json", "show", "RollingStd");
  assert.equal(r.status, 0, r.stderr);
  assert.equal(JSON.parse(r.stdout).name, "RollingStd");
  r = run("show", "NoSuchThing");
  assert.equal(r.status, 1);
  assert.match(r.stderr, /no function named/);
  r = run("topics");
  assert.equal(r.status, 0, r.stderr);
  assert.match(r.stdout, /^Statistics\n/);
});
