// Search and read screamer's built-in documentation, from JavaScript.
//
//   import { search, show, topics, guide } from "@screamer-labs/screamer/docs";
//   search("order flow imbalance");   // [{ name, kind, short, topics }, ...]
//   show("RollingRSI");               // parameters, signatures, details, ...
//
// The same calls run from a shell as `npx screamer-docs search <keywords>`.
// The documentation is the one the Python package ships (screamer/data),
// translated to JavaScript names by devtools/wasm/gen_js_docs.py: functions
// are camelCase (`positionTest`), stream operators are lowercase functions
// (`combineLatest`), and parameters are camelCase options. It is a separate
// entry point, so it adds nothing to a bundle that imports only the operators.
import { DOCS_JSON } from "./generated/docs_data.js";

export const DOCS_URL = "https://screamer.readthedocs.io/en/latest/";

/** One parameter of a functor. */
export interface DocParameter {
  name: string;
  type?: string;
  /** The default as written in JavaScript, such as `14`, `"wilder"`, or `-Infinity`. */
  defaultText?: string;
  enum?: unknown[];
  min?: number | string;
  max?: number | string;
  description?: string;
  [key: string]: unknown;
}

/** The full documentation of one function, class, or guide. */
export interface DocEntry {
  name: string;
  /** The name in the Python package, such as `position_test`. */
  pythonName: string;
  kind: string;
  title: string;
  short: string;
  topics: string[];
  tags: string[];
  /** Call signatures: one per TypeScript overload, or the options form of a functor. */
  signatures?: string[];
  parameters?: DocParameter[];
  inputs?: number | string;
  outputs?: number | string;
  outputNames?: string[];
  nanPolicy?: string;
  seeAlso: string[];
  /** Description in markdown. */
  details?: string;
  /** Examples; they are in Python, with the same arguments as in JavaScript. */
  examples?: { language?: string; caption?: string; code: string }[];
  url?: string;
}

/** One search result. */
export interface DocSummary { name: string; kind: string; short: string; topics: string[] }

/** A topic group, its topics, and the functions in each topic. */
export interface DocGroup {
  slug: string;
  name: string;
  desc: string;
  topics: { slug: string; name: string; desc: string; functions: string[] }[];
}

interface DocsData {
  entries: Record<string, DocEntry & { index: boolean }>;
  groups: { slug: string; name: string; desc: string; topics: { slug: string; name: string; desc: string }[] }[];
  guide: string;
}

let data: DocsData | null = null;
function docs(): DocsData {
  return (data ??= JSON.parse(DOCS_JSON) as DocsData);
}

function summary(e: DocEntry): DocSummary {
  return { name: e.name, kind: e.kind, short: e.short, topics: e.topics };
}

/** The topic groups in display order, each topic listing its functions. */
export function topics(): DocGroup[] {
  const byTopic = new Map<string, string[]>();
  for (const e of Object.values(docs().entries)) {
    if (!e.index) continue;
    for (const t of e.topics) {
      if (!byTopic.has(t)) byTopic.set(t, []);
      byTopic.get(t)!.push(e.name);
    }
  }
  const lower = (s: string) => s.toLowerCase();
  return docs().groups.map((g) => ({
    ...g,
    topics: g.topics.map((t) => ({
      ...t,
      functions: [...(byTopic.get(t.slug) ?? [])].sort((a, b) =>
        lower(a) < lower(b) ? -1 : lower(a) > lower(b) ? 1 : 0),
    })),
  }));
}

// Search works exactly as in Python (screamer/docs.py), so both rank alike.
function words(text: string): string[] {
  // A trailing "s" is dropped so that "bands" finds "band" and the reverse.
  return (text.toLowerCase().match(/[a-z0-9]+/g) ?? []).map((w) =>
    w.length > 3 && w.endsWith("s") ? w.slice(0, -1) : w);
}

type Field = [Set<string>, number];
let index: [DocEntry, Field[]][] | null = null;
function searchIndex(): [DocEntry, Field[]][] {
  if (index) return index;
  index = Object.values(docs().entries).map((e) => {
    const name = e.pythonName;
    // Split CamelCase and snake_case names into their words too.
    const nameWords = (name.match(/[A-Z]+(?![a-z])|[A-Z]?[a-z]+|\d+/g) ?? []).join(" ");
    const fields: Field[] = [
      [new Set([...words(name), ...words(nameWords)]), 6],
      [new Set(words(e.tags.join(" "))), 4],
      [new Set(words(`${e.title} ${e.short}`)), 3],
      [new Set(words(e.topics.join(" "))), 2],
      [new Set(words(e.details ?? "")), 1],
    ];
    return [e, fields];
  });
  return index;
}

/**
 * Functions that match `query`, best match first.
 *
 * Each word of the query is looked up in a function's name, tags, title and
 * summary, topics, and description, in that order of weight, and a word found
 * in several of them counts for each. A function that matches more of the
 * words ranks higher, and a name equal to the query ranks first. `topic` keeps
 * only the functions in that topic slug (see {@link topics}).
 */
export function search(query: string, opts: { topic?: string; limit?: number } = {}): DocSummary[] {
  const { topic, limit = 10 } = opts;
  const qwords = words(query);
  const exact = query.trim().toLowerCase();
  const scored: [number, string, DocEntry][] = [];
  for (const [e, fields] of searchIndex()) {
    if (topic !== undefined && topic !== null && !e.topics.includes(topic)) continue;
    let hits = 0, score = 0;
    for (const w of qwords) {
      let weight = 0;
      for (const [ws, wt] of fields) if (ws.has(w)) weight += wt;
      if (weight > 0) hits++;
      score += weight;
    }
    if (e.name.toLowerCase() === exact || e.pythonName.toLowerCase() === exact) score += 100;
    if (score) scored.push([-(hits * 10 + score), e.pythonName.toLowerCase(), e]);
  }
  scored.sort((a, b) => a[0] - b[0] || (a[1] < b[1] ? -1 : a[1] > b[1] ? 1 : 0));
  return scored.slice(0, limit).map(([, , e]) => summary(e));
}

function similarity(a: string, b: string): number {
  // 1 - edit distance / longer length, where swapping two neighbouring
  // letters counts as one edit.
  const m = a.length, n = b.length;
  const d = Array.from({ length: m + 1 }, (_, i) => Array.from({ length: n + 1 }, (_, j) => (i ? (j ? 0 : i) : j)));
  for (let i = 1; i <= m; i++)
    for (let j = 1; j <= n; j++) {
      const cost = a[i - 1] === b[j - 1] ? 0 : 1;
      d[i][j] = Math.min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + cost);
      if (i > 1 && j > 1 && a[i - 1] === b[j - 2] && a[i - 2] === b[j - 1])
        d[i][j] = Math.min(d[i][j], d[i - 2][j - 2] + 1);
    }
  return 1 - d[m][n] / Math.max(m, n, 1);
}

/**
 * The full documentation of one function, by its JavaScript or Python name.
 * An unknown name throws an Error that names the closest matches.
 */
export function show(name: string): DocEntry {
  const entries = docs().entries;
  const all = Object.values(entries);
  const lower = name.toLowerCase();
  const e = entries[name]
    ?? all.find((x) => x.pythonName === name)
    ?? all.find((x) => x.name.toLowerCase() === lower || x.pythonName.toLowerCase() === lower);
  if (!e) {
    const close = all
      .map((x) => [similarity(lower, x.name.toLowerCase()), x.name] as const)
      .filter(([s]) => s >= 0.5)
      .sort((a, b) => b[0] - a[0])
      .slice(0, 5)
      .map(([, n]) => n);
    throw new Error(`no function named '${name}'` + (close.length ? `; did you mean ${close.join(", ")}?` : ""));
  }
  const { index: _index, ...rest } = e;
  return rest;
}

/** How to use screamer, written for coding agents (markdown). */
export function guide(): string {
  return docs().guide;
}

// --------------------------------------------------------------------------
// Text output, for the command line and for agents
// --------------------------------------------------------------------------

/** {@link topics} as text. */
export function formatTopics(groups: DocGroup[] = topics()): string {
  const lines: string[] = [];
  for (const g of groups) {
    lines.push(g.name, `  ${g.desc}`);
    for (const t of g.topics) lines.push(`  - ${t.slug}: ${t.functions.join(", ")}`);
    lines.push("");
  }
  return lines.join("\n").trimEnd() + "\n";
}

/** {@link search} results as text. */
export function formatSearch(results: DocSummary[]): string {
  if (!results.length) return "No matches.\n";
  const width = Math.max(...results.map((r) => r.name.length));
  return results.map((r) => `${r.name.padEnd(width)}  ${r.short}\n`).join("");
}

/** {@link show} as markdown text. */
export function formatShow(e: DocEntry): string {
  const lines = [`# ${e.name}`, "", e.short, ""];
  if (e.signatures?.length) {
    const lang = e.parameters ? "javascript" : "typescript";
    lines.push("```" + lang, ...e.signatures, "```", "");
  }
  if (e.parameters?.length) {
    lines.push("## Parameters", "");
    for (const p of e.parameters) {
      const extra: string[] = [];
      if (p.type) extra.push(p.type);
      if (p.defaultText !== undefined) extra.push(`default ${p.defaultText}`);
      if (p.enum) extra.push("one of " + p.enum.map((v) => JSON.stringify(v)).join(", "));
      if (p.min !== undefined) extra.push(`min ${p.min}`);
      if (p.max !== undefined) extra.push(`max ${p.max}`);
      lines.push(`- \`${p.name}\` (${extra.join("; ")}): ${p.description ?? ""}`);
    }
    lines.push("");
  }
  const facts: string[] = [];
  if (e.inputs !== undefined) facts.push(`Inputs: ${e.inputs}`);
  if (e.outputs !== undefined)
    facts.push(`Outputs: ${e.outputs}` + (e.outputNames ? ` (${e.outputNames.join(", ")})` : ""));
  if (e.nanPolicy) facts.push(`NaN policy: ${e.nanPolicy}`);
  if (e.topics.length) facts.push(`Topics: ${e.topics.join(", ")}`);
  if (e.seeAlso.length) facts.push(`See also: ${e.seeAlso.join(", ")}`);
  if (e.pythonName !== e.name) facts.push(`Python name: ${e.pythonName}`);
  if (e.url) facts.push(`Documentation: ${e.url}`);
  lines.push(...facts, "");
  if (e.details) lines.push(e.details.trim(), "");
  for (const ex of e.examples ?? []) {
    lines.push(`## Example in Python: ${ex.caption || e.name}`, "",
      "```" + (ex.language ?? "python"), ex.code.trimEnd(), "```", "");
  }
  return lines.join("\n").trimEnd() + "\n";
}
