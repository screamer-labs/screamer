#!/usr/bin/env node
// screamer-docs: search and read screamer's documentation from a shell.
//
//   npx screamer-docs search order flow imbalance
//   npx screamer-docs show RollingRSI
//   npx screamer-docs topics
//   npx screamer-docs guide
//
// Add --json for structured output.
import { formatSearch, formatShow, formatTopics, guide, search, show, topics } from "./docs.js";

const USAGE = `usage: screamer-docs [--json] <command> [arguments]

Search and read screamer's documentation (the installed version).

commands:
  topics                    list the topic groups and their functions
  search <keywords...>      find functions by keywords
         [--topic <slug>] [--limit <n>]
  show <Name>               the full documentation of one function
  guide                     how to use screamer, written for coding agents
`;

function main(argv: string[]): number {
  const args = [...argv];
  const take = (flag: string): string | undefined => {
    const i = args.indexOf(flag);
    if (i === -1) return undefined;
    const v = args[i + 1];
    args.splice(i, 2);
    return v;
  };
  const json = args.includes("--json");
  if (json) args.splice(args.indexOf("--json"), 1);
  if (args.includes("--help") || args.includes("-h") || !args.length) {
    process.stdout.write(USAGE);
    return args.length ? 0 : 2;
  }
  const topic = take("--topic");
  const limitText = take("--limit");
  const [command, ...rest] = args;
  const out = (value: unknown, text: string) =>
    process.stdout.write(json ? JSON.stringify(value, null, 2) + "\n" : text);

  switch (command) {
    case "topics": {
      const groups = topics();
      out(groups, formatTopics(groups));
      return 0;
    }
    case "search": {
      if (!rest.length) break;
      const limit = limitText === undefined ? 10 : Number(limitText);
      const results = search(rest.join(" "), { topic, limit });
      out(results, formatSearch(results));
      return 0;
    }
    case "show": {
      if (rest.length !== 1) break;
      try {
        const e = show(rest[0]);
        out(e, formatShow(e));
        return 0;
      } catch (err) {
        process.stderr.write(`${(err as Error).message}\n`);
        return 1;
      }
    }
    case "guide":
      out(guide(), guide());
      return 0;
  }
  process.stderr.write(USAGE);
  return 2;
}

process.exitCode = main(process.argv.slice(2));
